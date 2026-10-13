use super::*;
use spatialrust_image::{Image, ImageMetadata, ImageRegion, ImageView};
use spatialrust_vision::{canny, CannyOptions};
use std::collections::VecDeque;

fn cpu_options(o: GpuCannyOptions) -> CannyOptions {
    CannyOptions {
        low_threshold: o.low_threshold,
        high_threshold: o.high_threshold,
        aperture_size: o.aperture_size as usize,
        l2_gradient: o.l2_gradient,
    }
}

fn fixture(w: usize, h: usize, profile: u32) -> Image<u8, 1> {
    let data = (0..w * h)
        .map(|i| {
            let x = i % w;
            let y = i / w;
            match profile {
                0 => 0,
                1 => ((i * 7919 + y * 31 + 17) % 256) as u8,
                2 => {
                    if (x / 7 + y / 11) % 2 == 0 {
                        230
                    } else {
                        19
                    }
                }
                _ => ((x * 7 + y * 13 + (x / 9) * 31) % 96) as u8,
            }
        })
        .collect();
    Image::try_new(w, h, data).unwrap()
}

#[path = "canny_frozen.rs"]
mod frozen;
use frozen::*;

#[test]
fn all_apertures_norms_thresholds_and_strips_match_cpu_edge_bytes() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let previous = previous_pipelines(&runtime);
    let horizontal_previous = previous_horizontal_pipelines(&runtime);
    let state_previous = previous_state_flags_pipelines(&runtime);
    let rows_previous = previous_reuse_rows_pipelines(&runtime);
    let (mut calls, mut bytes) = (0, 0);
    for (w, h) in [(1, 1), (1, 37), (37, 1), (2, 3), (17, 19), (65, 33)] {
        for profile in 0..4 {
            let image = fixture(w, h, profile);
            let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
            for aperture_size in [3, 5, 7] {
                for l2_gradient in [false, true] {
                    for (low_threshold, high_threshold) in [
                        (0.0, 0.0),
                        (32.5, 70.75),
                        (100.0, 200.0),
                        (1020.0, 1020.0),
                        (200.0, 100.0),
                        (f64::MAX, f64::MAX),
                    ] {
                        let o = GpuCannyOptions {
                            aperture_size,
                            l2_gradient,
                            low_threshold,
                            high_threshold,
                        };
                        let expected = canny(image.view(), cpu_options(o)).unwrap();
                        let mut result = canny_gpu(&runtime, &source, o).unwrap();
                        assert_eq!(result.metadata(), image.metadata());
                        assert_eq!(result.channels(), 1);
                        result.receipt().validate_resident_chain((w * h * 4) as u64).unwrap();
                        assert_eq!(
                            result.receipt().gpu_to_gpu_bytes(),
                            (w * h * if aperture_size == 3 { 32 } else { 36 }) as u64
                        );
                        assert_eq!(
                            result.readback_u8::<1>(&runtime).unwrap(),
                            expected,
                            "{w}x{h} profile={profile} {o:?}"
                        );
                        result.recycle(&runtime);
                        let mut old = previous_canny(&runtime, &source, o, &previous);
                        assert_eq!(
                            old.readback_u8::<1>(&runtime).unwrap(),
                            expected,
                            "unpacked {w}x{h} profile={profile} {o:?}"
                        );
                        old.recycle(&runtime);
                        let mut horizontal_old =
                            previous_horizontal_canny(&runtime, &source, o, &horizontal_previous);
                        assert_eq!(
                            horizontal_old.readback_u8::<1>(&runtime).unwrap(),
                            expected,
                            "horizontal {w}x{h} profile={profile} {o:?}"
                        );
                        horizontal_old.recycle(&runtime);
                        let mut state_old =
                            previous_state_flags_canny(&runtime, &source, o, &state_previous);
                        assert_eq!(
                            state_old.readback_u8::<1>(&runtime).unwrap(),
                            expected,
                            "state flags {w}x{h} profile={profile} {o:?}"
                        );
                        assert_eq!(
                            state_old.receipt().gpu_to_gpu_bytes(),
                            (w * h * if aperture_size == 3 { 36 } else { 40 }) as u64
                        );
                        state_old.recycle(&runtime);
                        let mut row_old =
                            previous_reuse_rows_canny(&runtime, &source, o, &rows_previous);
                        assert_eq!(
                            row_old.readback_u8::<1>(&runtime).unwrap(),
                            expected,
                            "reuse rows {w}x{h} profile={profile} {o:?}"
                        );
                        assert_eq!(
                            row_old.receipt().gpu_to_gpu_bytes(),
                            (w * h * if aperture_size == 3 { 32 } else { 36 }) as u64
                        );
                        row_old.recycle(&runtime);
                        calls += 1;
                        bytes += w * h;
                    }
                }
            }
            source.recycle(&runtime);
        }
    }
    eprintln!("GPU_CANNY_CPU calls={calls} exact_edge_bytes={bytes}");
}

struct Snapshot {
    gradients: Vec<[i32; 4]>,
    horizontal: Vec<[i32; 2]>,
    state_words: Vec<u32>,
    states: Vec<u32>,
    roots: Vec<u32>,
    flags: Vec<u32>,
    edges: Image<u8, 1>,
}

// Encode copies while all submitted scratch is still exclusively leased.
fn snapshot(
    runtime: &WgpuRuntime,
    image: &Image<u8, 1>,
    o: GpuCannyOptions,
    mask: Option<&[u32]>,
    force_one_group: bool,
) -> Snapshot {
    snapshot_with_cache(runtime, image, o, mask, force_one_group, None)
}

fn snapshot_with_cache(
    runtime: &WgpuRuntime,
    image: &Image<u8, 1>,
    o: GpuCannyOptions,
    mask: Option<&[u32]>,
    force_one_group: bool,
    previous: Option<(&CannyPipelines, FrozenLayout)>,
) -> Snapshot {
    let source = GpuImage::upload_u8(runtime, image.view()).unwrap();
    let mut p = plan(source.width(), source.height(), 1, o, &runtime.device().limits()).unwrap();
    if force_one_group {
        p.params.groups = 1;
    }
    if let Some((_, layout)) = previous {
        frozen_plan(&mut p, layout);
    }
    let horizontal_bytes =
        if o.aperture_size != 3 && mask.is_none() { p.horizontal_bytes } else { 0 };
    let cache = previous.map(|(cache, _)| cache).unwrap_or_else(|| {
        runtime.image_canny_pipelines.get_or_init(|| CannyPipelines::new(runtime.device()))
    });
    let mut scratch = ScratchBuffers::new(runtime);
    let work = Work::acquire(&mut scratch, &p);
    let horizontal_buffer =
        previous.map(|_| scratch.acquire("canny-horizontal", p.horizontal_bytes));
    let separate_flags = previous.is_some_and(|(_, layout)| layout.separate_flags());
    let strong = separate_flags.then(|| scratch.acquire("canny-strong-roots", p.pixel_bytes));
    runtime.queue().write_buffer(&work.states, 0, &vec![0xff; p.pixel_bytes as usize]);
    let output = create_texture(runtime, source.width(), source.height(), "canny-test-output");
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let group = if let Some(horizontal) = &horizontal_buffer {
        bind_legacy(
            runtime,
            cache,
            &work,
            (horizontal, strong.as_ref()),
            &uniform,
            &source.view(),
            &output.create_view(&Default::default()),
        )
    } else {
        work.bind(
            runtime,
            cache,
            &uniform,
            &source.view(),
            &output.create_view(&Default::default()),
        )
    };
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: p.gradient_bytes + 3 * p.pixel_bytes + horizontal_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    if let Some(mask) = mask {
        assert_eq!(mask.len(), p.params.pixels as usize);
        runtime.queue().write_buffer(&work.gradients, 0, &vec![0; p.gradient_bytes as usize]);
        encode(cache, &group, p.params.groups, &[3], &mut encoder, None);
        let uploaded = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(mask),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        encoder.copy_buffer_to_buffer(&uploaded, 0, &work.states, 0, p.pixel_bytes);
        encode(cache, &group, p.params.groups, &[4, 5, 6], &mut encoder, None);
    } else if previous.is_none() && o.aperture_size != 3 {
        // The row lifetime ends before classification overwrites this binding.
        // This test-only copy records exact unrounded rows while still leased.
        encode(cache, &group, p.params.groups, &[1, 2], &mut encoder, None);
        encoder.copy_buffer_to_buffer(
            &work.states,
            0,
            &staging,
            p.gradient_bytes + 3 * p.pixel_bytes,
            horizontal_bytes,
        );
        encode(cache, &group, p.params.groups, &[3, 4, 5, 6], &mut encoder, None);
    } else {
        encode(cache, &group, p.params.groups, pass_indices(o.aperture_size), &mut encoder, None);
    }
    for (buffer, offset, bytes) in [
        (&work.gradients, 0, p.gradient_bytes),
        (&work.states, p.gradient_bytes, p.pixel_bytes),
        (&work.parent, p.gradient_bytes + p.pixel_bytes, p.pixel_bytes),
        (
            strong.as_ref().unwrap_or(&work.states),
            p.gradient_bytes + 2 * p.pixel_bytes,
            p.pixel_bytes,
        ),
    ] {
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, offset, bytes);
    }
    if let Some(horizontal) = horizontal_buffer.as_ref().filter(|_| horizontal_bytes != 0) {
        encoder.copy_buffer_to_buffer(
            horizontal,
            0,
            &staging,
            p.gradient_bytes + 3 * p.pixel_bytes,
            horizontal_bytes,
        );
    }
    scratch.submit_and_recycle(encoder);
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    runtime.device().poll(wgpu::Maintain::Wait);
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let words = bytemuck::cast_slice::<u8, u32>(&mapped);
    let n = p.params.pixels as usize;
    let gradient_words = p.gradient_bytes as usize / 4;
    let gradients = if previous.is_some_and(|(_, layout)| layout == FrozenLayout::Unpacked) {
        words[..gradient_words]
            .chunks_exact(4)
            .map(|v| [v[0] as i32, v[1] as i32, v[2] as i32, v[3] as i32])
            .collect()
    } else {
        words[..gradient_words]
            .chunks_exact(2)
            .map(|v| [i32::from(v[0] as i16), i32::from((v[0] >> 16) as i16), v[1] as i32, 0])
            .collect()
    };
    let state_words = words[gradient_words..gradient_words + n].to_vec();
    let states =
        state_words.iter().map(|&word| if separate_flags { word } else { word & 3 }).collect();
    let roots = words[gradient_words + n..gradient_words + 2 * n].to_vec();
    let flags = words[gradient_words + 2 * n..gradient_words + 3 * n]
        .iter()
        .map(|&word| if separate_flags { word } else { (word >> 2) & 1 })
        .collect();
    let horizontal_words = &words[gradient_words + 3 * n..];
    let horizontal = if previous.is_some_and(|(_, layout)| !layout.packed_rows()) {
        horizontal_words.chunks_exact(2).map(|v| [v[0] as i32, v[1] as i32]).collect()
    } else {
        horizontal_words
            .iter()
            .map(|&v| [i32::from(v as i16), i32::from((v >> 16) as i16)])
            .collect()
    };
    drop(mapped);
    staging.unmap();
    let mut output = GpuImage::from_parts(
        runtime,
        source.width(),
        source.height(),
        1,
        output,
        source.metadata(),
        Default::default(),
    )
    .unwrap();
    let edges = output.readback_u8::<1>(runtime).unwrap();
    output.recycle(runtime);
    source.recycle(runtime);
    Snapshot { gradients, horizontal, state_words, states, roots, flags, edges }
}

fn choose(n: usize, k: isize) -> i64 {
    if k < 0 || k as usize > n {
        return 0;
    }
    (0..k as usize).fold(1, |v, i| v * (n - i) as i64 / (i + 1) as i64)
}

// Independent unrounded 1D binomial derivative/smoothing at replicated borders.
fn reference_horizontal(image: &Image<u8, 1>, o: GpuCannyOptions) -> Vec<[i32; 2]> {
    let size = o.aperture_size as usize;
    if size == 3 {
        return vec![];
    }
    let (w, h) = (image.width(), image.height());
    (0..w * h)
        .map(|index| {
            let x = (index % w) as isize;
            let y = index / w;
            let (mut derivative, mut smooth) = (0i64, 0i64);
            for tap in 0..size {
                let sx = (x + tap as isize - (size / 2) as isize).clamp(0, w as isize - 1) as usize;
                let value = i64::from(image.as_slice()[y * w + sx]);
                derivative +=
                    value * (choose(size - 2, tap as isize - 1) - choose(size - 2, tap as isize));
                smooth += value * choose(size - 1, tap as isize);
            }
            [derivative as i32, smooth as i32]
        })
        .collect()
}

fn rounded_even(value: f64) -> i32 {
    let floor = value.floor();
    let remainder = value - floor;
    (floor as i32) + i32::from(remainder > 0.5 || (remainder == 0.5 && floor as i32 & 1 != 0))
}

// Independent direct 2D binary64 convolution, not the device separable passes.
fn reference_gradients(image: &Image<u8, 1>, o: GpuCannyOptions) -> Vec<[i32; 4]> {
    let size = o.aperture_size as usize;
    let radius = (size / 2) as isize;
    let (w, h) = (image.width(), image.height());
    (0..w * h)
        .map(|index| {
            let (x, y) = ((index % w) as isize, (index / w) as isize);
            let (mut gx, mut gy) = (0.0, 0.0);
            for j in 0..size {
                for i in 0..size {
                    let px = (x + i as isize - radius).clamp(0, w as isize - 1) as usize;
                    let py = (y + j as isize - radius).clamp(0, h as isize - 1) as usize;
                    let value = f64::from(image.as_slice()[py * w + px]);
                    gx += value
                        * (choose(size - 2, i as isize - 1) - choose(size - 2, i as isize)) as f64
                        * choose(size - 1, j as isize) as f64;
                    gy += value
                        * (choose(size - 2, j as isize - 1) - choose(size - 2, j as isize)) as f64
                        * choose(size - 1, i as isize) as f64;
                }
            }
            if size == 7 {
                gx /= 16.0;
                gy /= 16.0;
            }
            let (gx, gy) = (rounded_even(gx), rounded_even(gy));
            let mag = if o.l2_gradient {
                i64::from(gx) * i64::from(gx) + i64::from(gy) * i64::from(gy)
            } else {
                i64::from(gx.abs() + gy.abs())
            };
            [gx, gy, mag as i32, 0]
        })
        .collect()
}

fn reference_states(w: usize, h: usize, gradients: &[[i32; 4]], o: GpuCannyOptions) -> Vec<u32> {
    let (mut low, mut high) = (o.low_threshold, o.high_threshold);
    if o.aperture_size == 7 {
        low /= 16.0;
        high /= 16.0;
    }
    if low > high {
        std::mem::swap(&mut low, &mut high);
    }
    if o.l2_gradient {
        low = low.min(32767.0).powi(2);
        high = high.min(32767.0).powi(2);
    }
    let (low, high) = (low.floor() as i64, high.floor() as i64);
    (0..w * h)
        .map(|index| {
            let g = gradients[index];
            let (x, y) = ((index % w) as isize, (index / w) as isize);
            let mag = i64::from(g[2]);
            let neighbor = |dx: isize, dy: isize| {
                let (x, y) = (x + dx, y + dy);
                if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
                    0
                } else {
                    i64::from(gradients[y as usize * w + x as usize][2])
                }
            };
            let ax = i64::from(g[0].abs());
            let ay = i64::from(g[1].abs()) * 32768;
            let maximum = if ay < ax * 13573 {
                mag > neighbor(-1, 0) && mag >= neighbor(1, 0)
            } else if ay > ax * 13573 + ax * 65536 {
                mag > neighbor(0, -1) && mag >= neighbor(0, 1)
            } else {
                let sign =
                    if g[0].signum() != g[1].signum() && g[0] != 0 && g[1] != 0 { -1 } else { 1 };
                mag > neighbor(-sign, -1) && mag > neighbor(sign, 1)
            };
            if mag <= low || !maximum {
                0
            } else if mag > high {
                2
            } else {
                1
            }
        })
        .collect()
}

// Independent BFS components also prove smallest-index root and strong flags.
fn reference_components(w: usize, h: usize, states: &[u32]) -> (Vec<u32>, Vec<u32>, Vec<u8>) {
    let mut roots: Vec<_> = (0..states.len() as u32).collect();
    let mut flags = vec![0; states.len()];
    let mut edges: Vec<u8> = states.iter().map(|&state| if state == 2 { 255 } else { 0 }).collect();
    let mut seen = vec![false; states.len()];
    let mut queue = VecDeque::new();
    for start in 0..states.len() {
        if seen[start] || states[start] != 1 {
            continue;
        }
        let mut component = vec![];
        let mut strong = false;
        seen[start] = true;
        queue.push_back(start);
        while let Some(index) = queue.pop_front() {
            component.push(index);
            let (x, y) = (index % w, index / w);
            for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    let neighbor = ny * w + nx;
                    strong |= states[neighbor] == 2;
                    if !seen[neighbor] && states[neighbor] == 1 {
                        seen[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        let root = *component.iter().min().unwrap();
        flags[root] = u32::from(strong);
        for i in component {
            roots[i] = root as u32;
            edges[i] = if strong { 255 } else { 0 };
        }
    }
    (roots, flags, edges)
}

#[test]
fn owned_snapshots_match_direct_f64_gradients_nms_and_bfs_roots() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let previous = previous_pipelines(&runtime);
    let horizontal_previous = previous_horizontal_pipelines(&runtime);
    for profile in 0..4 {
        let image = fixture(97, 53, profile);
        for aperture_size in [3, 5, 7] {
            for l2_gradient in [false, true] {
                let o = GpuCannyOptions {
                    aperture_size,
                    l2_gradient,
                    low_threshold: 40.25,
                    high_threshold: 120.75,
                };
                let observed = snapshot(&runtime, &image, o, None, true);
                let gradients = reference_gradients(&image, o);
                let states = reference_states(97, 53, &gradients, o);
                let (roots, flags, edges) = reference_components(97, 53, &states);
                assert_eq!(observed.gradients, gradients, "profile={profile} {o:?}");
                assert_eq!(observed.states, states);
                assert_eq!(observed.roots, roots);
                assert_eq!(observed.flags, flags);
                assert_eq!(observed.edges.as_slice(), edges);
                let old = snapshot_with_cache(
                    &runtime,
                    &image,
                    o,
                    None,
                    true,
                    Some((&previous, FrozenLayout::Unpacked)),
                );
                let horizontal = reference_horizontal(&image, o);
                assert_eq!(observed.horizontal, horizontal);
                assert_eq!(old.horizontal, horizontal);
                assert_eq!(old.gradients, observed.gradients);
                assert_eq!(old.states, observed.states);
                assert_eq!(old.roots, observed.roots);
                assert_eq!(old.flags, observed.flags);
                assert_eq!(old.edges, observed.edges);
                let horizontal_old = snapshot_with_cache(
                    &runtime,
                    &image,
                    o,
                    None,
                    true,
                    Some((&horizontal_previous, FrozenLayout::PackedGradients)),
                );
                assert_eq!(horizontal_old.horizontal, horizontal);
                assert_eq!(horizontal_old.gradients, observed.gradients);
                assert_eq!(horizontal_old.states, observed.states);
                assert_eq!(horizontal_old.roots, observed.roots);
                assert_eq!(horizontal_old.flags, observed.flags);
                assert_eq!(horizontal_old.edges, observed.edges);
            }
        }
    }
}

#[test]
fn hysteresis_matches_bfs_for_long_weak_snakes_diagonals_and_disconnected_components() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut cases = vec![];
    for (w, h) in [(1, 1), (65, 1), (1, 65), (67, 71)] {
        for profile in 0..4 {
            let mut states: Vec<_> = (0..w * h)
                .map(|i| match profile {
                    0 => 0,
                    1 => 1,
                    2 => u32::from((i % w + i / w) % 5 == 0),
                    _ => u32::from(i % w == i / w % w),
                })
                .collect();
            if profile >= 1 {
                states[w * h - 1] = 2;
            }
            cases.push((w, h, states));
        }
    }
    let (w, h) = (1025, 1025);
    let mut snake = vec![0; w * h];
    for y in 0..h {
        if y % 3 == 0 {
            snake[y * w..(y + 1) * w].fill(1);
        } else {
            let x = if (y / 3) % 2 == 0 { w - 1 } else { 0 };
            snake[y * w + x] = 1;
        }
    }
    snake[w * h - 1] = 2;
    cases.push((w, h, snake.clone()));
    snake[(h / 2 / 3 * 3) * w + w / 2] = 0;
    cases.push((w, h, snake));
    for (w, h, states) in cases {
        let image = Image::try_new(w, h, vec![0; w * h]).unwrap();
        let observed = snapshot(&runtime, &image, GpuCannyOptions::default(), Some(&states), false);
        let (roots, flags, edges) = reference_components(w, h, &states);
        assert_eq!(observed.states, states);
        assert_eq!(observed.roots, roots, "{w}x{h}");
        assert_eq!(observed.flags, flags);
        assert_eq!(observed.edges.as_slice(), edges);
    }
}

#[test]
fn plan_and_invalid_public_inputs_reject_before_resource_initialization() {
    let limits = wgpu::Limits::default();
    let o = GpuCannyOptions::default();
    assert_eq!(std::mem::size_of::<Params>(), 96);
    for (w, h, c) in
        [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 2), (8193, 1, 1), (u32::MAX, u32::MAX, 1)]
    {
        assert!(plan(w, h, c, o, &limits).is_err());
    }
    for aperture_size in [0, 1, 2, 9, u32::MAX] {
        assert!(plan(1, 1, 1, GpuCannyOptions { aperture_size, ..o }, &limits).is_err());
    }
    for v in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(plan(1, 1, 1, GpuCannyOptions { low_threshold: v, ..o }, &limits).is_err());
        assert!(plan(1, 1, 1, GpuCannyOptions { high_threshold: v, ..o }, &limits).is_err());
    }
    for field in 0..14 {
        let mut bad = limits.clone();
        match field {
            0 => bad.max_buffer_size = 95,
            1 => bad.max_storage_buffer_binding_size = 7,
            2 => bad.max_uniform_buffer_binding_size = 95,
            3 => bad.max_compute_invocations_per_workgroup = 255,
            4 => bad.max_compute_workgroup_size_x = 255,
            5 => bad.max_compute_workgroup_size_y = 0,
            6 => bad.max_compute_workgroup_size_z = 0,
            7 => bad.max_compute_workgroups_per_dimension = 0,
            8 => bad.max_bind_groups = 0,
            9 => bad.max_bindings_per_bind_group = 7,
            10 => bad.max_storage_buffers_per_shader_stage = 2,
            11 => bad.max_uniform_buffers_per_shader_stage = 0,
            12 => bad.max_sampled_textures_per_shader_stage = 0,
            _ => bad.max_storage_textures_per_shader_stage = 0,
        }
        assert!(plan(1, 1, 1, o, &bad).is_err(), "field={field}");
    }
    let mut exact = limits.clone();
    exact.max_storage_buffer_binding_size = 8;
    exact.max_buffer_size = 96;
    assert!(plan(1, 1, 1, o, &exact).is_ok());
    assert!(plan(2, 1, 1, o, &exact).is_err());
    // An exact 4K binding now admits 4K and rejects an extra row. The tiny
    // binding also rejects a single pixel beyond capacity, before allocation.
    exact = limits.clone();
    exact.max_storage_buffer_binding_size = 3840 * 2160 * 8;
    for aperture_size in [3, 5, 7] {
        let o = GpuCannyOptions { aperture_size, ..o };
        let p = plan(3840, 2160, 1, o, &exact).unwrap();
        assert_eq!(p.gradient_bytes, 66_355_200);
        assert_eq!(p.gradient_bytes + 2 * p.pixel_bytes, 132_710_400);
        assert!(plan(3840, 2161, 1, o, &exact).is_err());
        exact.max_storage_buffer_binding_size = 40;
        assert!(plan(5, 1, 1, o, &exact).is_ok());
        assert!(plan(6, 1, 1, o, &exact).is_err());
        exact.max_storage_buffer_binding_size = 3840 * 2160 * 8;
    }
    let runtime = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let image = fixture(17, 19, 1);
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let color =
        GpuImage::upload_u8(&runtime, Image::<u8, 3>::try_new(1, 1, vec![1, 2, 3]).unwrap().view())
            .unwrap();
    let before = runtime.image_texture_pool_stats();
    let scratch = runtime.scratch_buffer_pool_stats();
    assert!(canny_gpu(&other, &source, o).is_err());
    assert!(canny_gpu(&runtime, &color, o).is_err());
    assert!(canny_gpu(&runtime, &source, GpuCannyOptions { low_threshold: f64::NAN, ..o }).is_err());
    assert_eq!(runtime.image_texture_pool_stats(), before);
    assert_eq!(runtime.scratch_buffer_pool_stats(), scratch);
    assert!(runtime.image_canny_pipelines.get().is_none());
}

#[test]
fn roi_metadata_concurrency_profiler_and_gaussian_ai_chain_keep_images_resident() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap();
    let other =
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap();
    let metadata =
        ImageMetadata { color_range: spatialrust_image::ColorRange::Limited, ..Default::default() };
    let data: Vec<_> = (0..83 * 79).map(|i| ((i * 31 + i / 83 * 13) % 256) as u8).collect();
    let view = ImageView::<u8, 1>::new_with_metadata(77, 79, 83, &data, metadata)
        .unwrap()
        .subview(ImageRegion::new(3, 5, 65, 67))
        .unwrap();
    let source = GpuImage::upload_u8(&runtime, view).unwrap();
    let o = GpuCannyOptions { aperture_size: 7, l2_gradient: true, ..Default::default() };
    let expected = canny(view, cpu_options(o)).unwrap();
    let before = runtime.image_texture_pool_stats();
    let scratch = runtime.scratch_buffer_pool_stats();
    if runtime.timestamp_queries_enabled() && other.timestamp_queries_enabled() {
        let mut short = GpuStageProfiler::new(&runtime, 5).unwrap();
        assert!(canny_gpu_profiled(&runtime, &source, o, &mut short).is_err());
        let mut wrong = GpuStageProfiler::new(&other, 6).unwrap();
        assert!(canny_gpu_profiled(&runtime, &source, o, &mut wrong).is_err());
        assert_eq!(runtime.image_texture_pool_stats(), before);
        assert_eq!(runtime.scratch_buffer_pool_stats(), scratch);
        assert!(runtime.image_canny_pipelines.get().is_none());
    }
    let mut profiler =
        runtime.timestamp_queries_enabled().then(|| GpuStageProfiler::new(&runtime, 6).unwrap());
    let profiled = if let Some(ref mut profiler) = profiler {
        canny_gpu_profiled(&runtime, &source, o, profiler).unwrap()
    } else {
        canny_gpu(&runtime, &source, o).unwrap()
    };
    let mut results = std::thread::scope(|scope| {
        let jobs: Vec<_> =
            (0..4).map(|_| scope.spawn(|| canny_gpu(&runtime, &source, o).unwrap())).collect();
        jobs.into_iter().map(|job| job.join().unwrap()).collect::<Vec<_>>()
    });
    results.push(profiled);
    source.recycle(&runtime);
    let runtime = Box::new(runtime);
    for output in &mut results {
        assert_eq!(output.metadata(), metadata);
        output.receipt().validate_resident_chain(65 * 67 * 4).unwrap();
        assert_eq!(output.readback_u8::<1>(&runtime).unwrap(), expected);
    }
    if let Some(profiler) = profiler {
        let report = profiler.collect(&runtime).unwrap();
        assert_eq!(report.device_to_host_bytes, 96);
        assert_eq!(
            report.stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
            [
                "canny_horizontal_gpu",
                "canny_gradients_gpu",
                "canny_classify_gpu",
                "canny_connect_gpu",
                "canny_roots_gpu",
                "canny_gpu"
            ]
        );
    }
    assert_eq!(runtime.initialized_image_pipeline_count(), 1);
    for output in results {
        output.recycle(&runtime);
    }
    let source = GpuImage::upload_u8(&runtime, view).unwrap();
    let allocations = runtime.scratch_buffer_pool_stats().allocation_count;
    for _ in 0..4 {
        let mut output = canny_gpu(&runtime, &source, o).unwrap();
        assert_eq!(output.readback_u8::<1>(&runtime).unwrap(), expected);
        output.recycle(&runtime);
    }
    assert_eq!(runtime.scratch_buffer_pool_stats().allocation_count, allocations);
    let blurred = crate::gaussian_blur_gpu(&runtime, &source, Default::default()).unwrap();
    let edges = canny_gpu(&runtime, &blurred, Default::default()).unwrap();
    let tensor = crate::pack_ai_chw_gpu(&runtime, &edges, 1.0 / 255.0, [0.0; 4], [1.0; 4]).unwrap();
    tensor.receipt().validate_resident_chain(65 * 67 * 4).unwrap();
    assert_eq!(
        tensor.receipt().stages(),
        [
            "upload_u8_texture",
            "gaussian_horizontal_gpu",
            "gaussian_blur_gpu",
            "canny_gradients_gpu",
            "canny_classify_gpu",
            "canny_connect_gpu",
            "canny_roots_gpu",
            "canny_gpu",
            "pack_ai_chw_gpu"
        ]
    );
}

// Probe the production pack/unpack/scale functions with signed host inputs.
// Readback copies are encoded while the scratch leases are still exclusive.
fn gradient_probe(
    runtime: &WgpuRuntime,
    values: &[[i32; 2]],
    count: usize,
    shader: &str,
) -> (Vec<i32>, Vec<u32>) {
    let (w, h) = (512, values.len().div_ceil(512));
    assert_eq!(values.len(), w * h);
    let image = Image::<u8, 1>::try_new(w, h, vec![0; w * h]).unwrap();
    let source = GpuImage::upload_u8(runtime, image.view()).unwrap();
    let mut p = plan(
        w as u32,
        h as u32,
        1,
        GpuCannyOptions { aperture_size: 7, ..Default::default() },
        &runtime.device().limits(),
    )
    .unwrap();
    p.params.pixels = count as u32;
    let cache = runtime.image_canny_pipelines.get_or_init(|| CannyPipelines::new(runtime.device()));
    let mut scratch = ScratchBuffers::new(runtime);
    let work = Work::acquire(&mut scratch, &p);
    runtime.queue().write_buffer(&work.gradients, 0, bytemuck::cast_slice(values));
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let output = create_texture(runtime, w as u32, h as u32, "gradient-probe-unused");
    let group = work.bind(
        runtime,
        cache,
        &uniform,
        &source.view(),
        &output.create_view(&Default::default()),
    );
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(format!("{SHADER}\n{shader}").into()),
    });
    let layout = runtime.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&cache.layout],
        push_constant_ranges: &[],
    });
    let pipeline = runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &module,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: p.gradient_bytes + p.pixel_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(p.params.groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&work.gradients, 0, &staging, 0, p.gradient_bytes);
    encoder.copy_buffer_to_buffer(&work.states, 0, &staging, p.gradient_bytes, p.pixel_bytes);
    scratch.submit_and_recycle(encoder);
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    runtime.device().poll(wgpu::Maintain::Wait);
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let words = bytemuck::cast_slice::<u8, u32>(&mapped);
    let gradients = words[..values.len() * 2].iter().map(|&v| v as i32).collect();
    let states = words[values.len() * 2..].to_vec();
    drop(mapped);
    staging.unmap();
    source.recycle(runtime);
    (gradients, states)
}

#[test]
fn aperture_seven_ties_even_rounding_matches_all_possible_raw_derivatives() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    const MAX_RAW: i32 = 163200;
    let count = (2 * MAX_RAW + 1) as usize;
    let values: Vec<[i32; 2]> =
        (0..count.div_ceil(512) * 512).map(|i| [i as i32 - MAX_RAW, 0]).collect();
    let shader = "@compute @workgroup_size(256) fn probe(@builtin(global_invocation_id) id:vec3<u32>) { for(var index=id.x;index<params.pixels;index=advance(index)) { gradients[index].magnitude=u32(scaled(i32(gradients[index].components))); atomicStore(&states[index],0u); } }";
    let (words, _) = gradient_probe(&runtime, &values, count, shader);
    for (i, value) in values.iter().take(count).enumerate() {
        assert_eq!(words[2 * i + 1], rounded_even(f64::from(value[0]) / 16.0), "raw index={i}");
    }
    eprintln!("GPU_CANNY_ROUND exact_signed_values={count}");
}

#[test]
fn packed_components_preserve_every_signed_sixteen_bit_value_and_word_layout() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let values: Vec<[i32; 2]> =
        (0..65536).map(|i| [i - 32768, ((i * 257 + 17) & 65535) - 32768]).collect();
    let shader = "@compute @workgroup_size(256) fn probe(@builtin(global_invocation_id) id:vec3<u32>) { for(var index=id.x;index<params.pixels;index=advance(index)) { let g=vec2<i32>(i32(gradients[index].components),i32(gradients[index].magnitude)); let packed=pack_components(g); let decoded=unpack_components(packed); gradients[index]=Gradient(u32(decoded.x),u32(decoded.y)); atomicStore(&states[index],packed); } }";
    let (words, packed) = gradient_probe(&runtime, &values, values.len(), shader);
    for (i, &[x, y]) in values.iter().enumerate() {
        assert_eq!(&words[2 * i..2 * i + 2], &[x, y], "signed index={i}");
        let expected = u32::from(x as u16) | (u32::from(y as u16) << 16);
        assert_eq!(packed[i], expected, "packed index={i}");
    }
    eprintln!("GPU_CANNY_PACK exact_signed_pairs={}", values.len());
}

#[test]
fn packed_gradient_bounds_are_reached_by_extremal_u8_kernels() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for (size, maximum) in [(3, 1020), (5, 12240), (7, 10200)] {
        for axis in [0, 1] {
            for sign in [-1, 1] {
                let data = (0..size * size)
                    .map(|i| {
                        let tap = if axis == 0 { i % size } else { i / size };
                        let coefficient =
                            choose(size - 2, tap as isize - 1) - choose(size - 2, tap as isize);
                        if coefficient * sign > 0 {
                            255
                        } else {
                            0
                        }
                    })
                    .collect();
                let image = Image::<u8, 1>::try_new(size, size, data).unwrap();
                for l2_gradient in [false, true] {
                    let o = GpuCannyOptions {
                        aperture_size: size as u32,
                        l2_gradient,
                        ..Default::default()
                    };
                    let expected = reference_gradients(&image, o);
                    let observed = snapshot(&runtime, &image, o, None, false);
                    assert_eq!(observed.gradients, expected);
                    assert_eq!(observed.gradients[(size * size) / 2][axis], maximum * sign as i32);
                }
            }
        }
    }
}

#[path = "canny_benchmark.rs"]
mod benchmark;
#[path = "canny_compact_benchmark.rs"]
mod compact_benchmark;

#[path = "canny_horizontal_tests.rs"]
mod horizontal_tests;

#[path = "canny_state_flags_tests.rs"]
mod state_flags_tests;

#[path = "canny_reuse_rows_tests.rs"]
mod reuse_rows_tests;
