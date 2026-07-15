//! Dense mask, depth, flow, confidence, and point-map primitives.

use std::collections::{BTreeMap, VecDeque};

use spatialrust_image::{ColorSpace, Image, ImageMetadata, ImageView};

use crate::{BoundingBox2, VisionError, VisionResult};

/// Pixel connectivity used by binary-mask algorithms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Connectivity {
    /// Horizontal and vertical neighbors.
    Four,
    /// Horizontal, vertical, and diagonal neighbors.
    #[default]
    Eight,
}

/// Validated binary mask (`0` background, `1` foreground).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryMask {
    image: Image<u8, 1>,
}

impl BinaryMask {
    /// Creates a mask and rejects values other than zero and one.
    pub fn try_new(width: usize, height: usize, data: Vec<u8>) -> VisionResult<Self> {
        if data.iter().any(|&value| value > 1) {
            return Err(VisionError::InvalidParameter(
                "binary mask values must be zero or one".to_owned(),
            ));
        }
        let metadata = ImageMetadata { color_space: ColorSpace::Label, ..Default::default() };
        Ok(Self { image: Image::try_new_with_metadata(width, height, data, metadata)? })
    }

    /// Thresholds a float score image into a binary mask.
    pub fn from_threshold(input: ImageView<'_, f32, 1>, threshold: f32) -> VisionResult<Self> {
        if !threshold.is_finite() {
            return Err(VisionError::InvalidParameter("mask threshold must be finite".to_owned()));
        }
        let mut data = Vec::with_capacity(input.width() * input.height());
        for y in 0..input.height() {
            for x in 0..input.width() {
                let value = input.get(x, y).expect("coordinate in bounds")[0];
                data.push(u8::from(value.is_finite() && value >= threshold));
            }
        }
        Self::try_new(input.width(), input.height(), data)
    }

    /// Returns mask width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.image.width()
    }

    /// Returns mask height.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.image.height()
    }

    /// Borrows the underlying image.
    #[must_use]
    pub fn image(&self) -> &Image<u8, 1> {
        &self.image
    }

    /// Borrows a mask view.
    #[must_use]
    pub fn view(&self) -> ImageView<'_, u8, 1> {
        self.image.view()
    }

    /// Returns whether one pixel is foreground.
    #[must_use]
    pub fn contains(&self, x: usize, y: usize) -> bool {
        self.image.get(x, y).is_some_and(|pixel| pixel[0] != 0)
    }

    /// Counts foreground pixels.
    #[must_use]
    pub fn area(&self) -> usize {
        self.image.as_slice().iter().filter(|&&value| value != 0).count()
    }

    /// Consumes the wrapper and returns its image.
    #[must_use]
    pub fn into_image(self) -> Image<u8, 1> {
        self.image
    }
}

/// Computes the exact Euclidean distance from every foreground pixel to the
/// nearest background pixel using unit pixel spacing.
///
/// Background pixels have distance zero. A non-empty mask must contain at
/// least one background pixel; otherwise the finite distance is undefined and
/// an error is returned. The implementation applies the separable linear-time
/// squared-distance transform by Felzenszwalb and Huttenlocher.
///
/// See <https://doi.org/10.4086/toc.2012.v008a019>.
pub fn distance_transform_edt(mask: &BinaryMask) -> VisionResult<Image<f32, 1>> {
    distance_transform_edt_with_spacing(mask, 1.0, 1.0)
}

/// Computes the exact Euclidean distance transform with physical pixel spacing.
///
/// `spacing_x` and `spacing_y` are the positive finite distances between
/// adjacent pixel centers along each axis. The output is expressed in those
/// physical units.
pub fn distance_transform_edt_with_spacing(
    mask: &BinaryMask,
    spacing_x: f32,
    spacing_y: f32,
) -> VisionResult<Image<f32, 1>> {
    if !spacing_x.is_finite() || spacing_x <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "distance-transform x spacing must be finite and positive".to_owned(),
        ));
    }
    if !spacing_y.is_finite() || spacing_y <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "distance-transform y spacing must be finite and positive".to_owned(),
        ));
    }

    let width = mask.width();
    let height = mask.height();
    let len = width.checked_mul(height).ok_or_else(|| {
        VisionError::InvalidDimensions("distance-transform size overflows".into())
    })?;
    if len == 0 {
        return Ok(Image::try_new_with_metadata(
            width,
            height,
            Vec::new(),
            ImageMetadata { color_space: ColorSpace::Gray, ..Default::default() },
        )?);
    }
    if mask.area() == len {
        return Err(VisionError::InvalidParameter(
            "distance transform requires at least one background pixel".to_owned(),
        ));
    }

    let mut horizontal = vec![f64::INFINITY; len];
    let mut source = vec![f64::INFINITY; width.max(height)];
    let mut transformed = vec![0.0_f64; width.max(height)];
    let mut sites = vec![0_usize; width.max(height)];
    let mut boundaries = vec![0.0_f64; width.max(height).saturating_add(1)];

    for y in 0..height {
        for (x, value) in source[..width].iter_mut().enumerate() {
            *value = if mask.contains(x, y) { f64::INFINITY } else { 0.0 };
        }
        squared_distance_transform_1d(
            &source[..width],
            f64::from(spacing_x).powi(2),
            &mut transformed[..width],
            &mut sites[..width],
            &mut boundaries[..=width],
        );
        horizontal[y * width..(y + 1) * width].copy_from_slice(&transformed[..width]);
    }

    let mut squared = vec![0.0_f64; len];
    for x in 0..width {
        for y in 0..height {
            source[y] = horizontal[y * width + x];
        }
        squared_distance_transform_1d(
            &source[..height],
            f64::from(spacing_y).powi(2),
            &mut transformed[..height],
            &mut sites[..height],
            &mut boundaries[..=height],
        );
        for y in 0..height {
            squared[y * width + x] = transformed[y];
        }
    }

    let distances = squared.into_iter().map(|value| value.sqrt() as f32).collect();
    Ok(Image::try_new_with_metadata(
        width,
        height,
        distances,
        ImageMetadata { color_space: ColorSpace::Gray, ..Default::default() },
    )?)
}

fn squared_distance_transform_1d(
    input: &[f64],
    coordinate_scale_squared: f64,
    output: &mut [f64],
    sites: &mut [usize],
    boundaries: &mut [f64],
) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert!(sites.len() >= input.len());
    debug_assert!(boundaries.len() > input.len());
    if input.is_empty() {
        return;
    }

    let Some(first) = input.iter().position(|value| value.is_finite()) else {
        output.fill(f64::INFINITY);
        return;
    };
    let mut envelope_end = 0_usize;
    sites[0] = first;
    boundaries[0] = f64::NEG_INFINITY;
    boundaries[1] = f64::INFINITY;

    for q in first + 1..input.len() {
        if !input[q].is_finite() {
            continue;
        }
        let mut intersection =
            parabola_intersection(input, coordinate_scale_squared, q, sites[envelope_end]);
        while envelope_end > 0 && intersection <= boundaries[envelope_end] {
            envelope_end -= 1;
            intersection =
                parabola_intersection(input, coordinate_scale_squared, q, sites[envelope_end]);
        }
        envelope_end += 1;
        sites[envelope_end] = q;
        boundaries[envelope_end] = intersection;
        boundaries[envelope_end + 1] = f64::INFINITY;
    }

    let mut envelope = 0_usize;
    for (q, value) in output.iter_mut().enumerate() {
        while boundaries[envelope + 1] < q as f64 {
            envelope += 1;
        }
        let site = sites[envelope];
        let delta = q as f64 - site as f64;
        *value = coordinate_scale_squared.mul_add(delta * delta, input[site]);
    }
}

fn parabola_intersection(
    input: &[f64],
    coordinate_scale_squared: f64,
    right: usize,
    left: usize,
) -> f64 {
    let right_coordinate = right as f64;
    let left_coordinate = left as f64;
    let right_height =
        coordinate_scale_squared.mul_add(right_coordinate * right_coordinate, input[right]);
    let left_height =
        coordinate_scale_squared.mul_add(left_coordinate * left_coordinate, input[left]);
    (right_height - left_height)
        / (2.0 * coordinate_scale_squared * (right_coordinate - left_coordinate))
}

/// Connected-component label image (`0` is background).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabelImage {
    image: Image<u32, 1>,
}

impl LabelImage {
    /// Returns label image width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.image.width()
    }

    /// Returns label image height.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.image.height()
    }

    /// Returns packed labels.
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        self.image.as_slice()
    }

    /// Returns one label.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> Option<u32> {
        self.image.get(x, y).map(|pixel| pixel[0])
    }

    /// Borrows the underlying image.
    #[must_use]
    pub fn image(&self) -> &Image<u32, 1> {
        &self.image
    }
}

/// Per-component statistics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComponentStats {
    /// Positive component label.
    pub label: u32,
    /// Foreground pixel count.
    pub area: usize,
    /// Half-open pixel bounding box.
    pub bbox: BoundingBox2,
    /// Mean pixel-center coordinate.
    pub centroid: [f64; 2],
}

/// Connected-component labeling output.
#[derive(Clone, Debug, PartialEq)]
pub struct ConnectedComponents {
    /// Per-pixel labels.
    pub labels: LabelImage,
    /// Statistics ordered by positive label.
    pub components: Vec<ComponentStats>,
}

/// Labels foreground components and computes bounding boxes and centroids.
pub fn connected_components(
    mask: &BinaryMask,
    connectivity: Connectivity,
) -> VisionResult<ConnectedComponents> {
    let width = mask.width();
    let height = mask.height();
    let mut labels = vec![0_u32; width * height];
    let mut components = Vec::new();
    let mut queue = VecDeque::new();
    let mut next_label = 1_u32;
    for seed_y in 0..height {
        for seed_x in 0..width {
            let seed = seed_y * width + seed_x;
            if !mask.contains(seed_x, seed_y) || labels[seed] != 0 {
                continue;
            }
            labels[seed] = next_label;
            queue.push_back((seed_x, seed_y));
            let mut area = 0_usize;
            let mut min_x = seed_x;
            let mut min_y = seed_y;
            let mut max_x = seed_x;
            let mut max_y = seed_y;
            let mut sum_x = 0.0_f64;
            let mut sum_y = 0.0_f64;
            while let Some((x, y)) = queue.pop_front() {
                area += 1;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                sum_x += x as f64 + 0.5;
                sum_y += y as f64 + 0.5;
                for (nx, ny) in neighbors(x, y, width, height, connectivity) {
                    let index = ny * width + nx;
                    if mask.contains(nx, ny) && labels[index] == 0 {
                        labels[index] = next_label;
                        queue.push_back((nx, ny));
                    }
                }
            }
            components.push(ComponentStats {
                label: next_label,
                area,
                bbox: BoundingBox2 {
                    x_min: min_x as f32,
                    y_min: min_y as f32,
                    x_max: (max_x + 1) as f32,
                    y_max: (max_y + 1) as f32,
                },
                centroid: [sum_x / area as f64, sum_y / area as f64],
            });
            next_label = next_label
                .checked_add(1)
                .ok_or_else(|| VisionError::InvalidDimensions("too many components".to_owned()))?;
        }
    }
    let metadata = ImageMetadata { color_space: ColorSpace::Label, ..Default::default() };
    let image = Image::try_new_with_metadata(width, height, labels, metadata)?;
    Ok(ConnectedComponents { labels: LabelImage { image }, components })
}

fn neighbors(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    connectivity: Connectivity,
) -> impl Iterator<Item = (usize, usize)> {
    let offsets: &[(isize, isize)] = match connectivity {
        Connectivity::Four => &[(0, -1), (-1, 0), (1, 0), (0, 1)],
        Connectivity::Eight => {
            &[(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        }
    };
    offsets.iter().filter_map(move |&(dx, dy)| {
        let nx = x.checked_add_signed(dx)?;
        let ny = y.checked_add_signed(dy)?;
        (nx < width && ny < height).then_some((nx, ny))
    })
}

/// One closed polygonal contour on pixel-grid corner coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Contour {
    /// Ordered vertices. The first vertex is not repeated at the end.
    pub points: Vec<[i32; 2]>,
}

/// Extracts oriented boundary loops, including hole contours.
pub fn find_contours(mask: &BinaryMask) -> Vec<Contour> {
    type Point = (i32, i32);
    let mut edges: BTreeMap<Point, Vec<Point>> = BTreeMap::new();
    let foreground = |x: isize, y: isize| {
        x >= 0
            && y >= 0
            && (x as usize) < mask.width()
            && (y as usize) < mask.height()
            && mask.contains(x as usize, y as usize)
    };
    for y in 0..mask.height() as isize {
        for x in 0..mask.width() as isize {
            if !foreground(x, y) {
                continue;
            }
            let x = x as i32;
            let y = y as i32;
            if !foreground(x as isize, y as isize - 1) {
                add_edge(&mut edges, (x, y), (x + 1, y));
            }
            if !foreground(x as isize + 1, y as isize) {
                add_edge(&mut edges, (x + 1, y), (x + 1, y + 1));
            }
            if !foreground(x as isize, y as isize + 1) {
                add_edge(&mut edges, (x + 1, y + 1), (x, y + 1));
            }
            if !foreground(x as isize - 1, y as isize) {
                add_edge(&mut edges, (x, y + 1), (x, y));
            }
        }
    }

    let mut contours = Vec::new();
    while let Some((&start, _)) = edges.iter().next() {
        let mut current = start;
        let mut points = Vec::new();
        loop {
            points.push([current.0, current.1]);
            let Some(next) = take_edge(&mut edges, current) else {
                break;
            };
            current = next;
            if current == start {
                break;
            }
        }
        if points.len() >= 3 {
            contours.push(Contour { points });
        }
    }
    contours
}

fn add_edge(edges: &mut BTreeMap<(i32, i32), Vec<(i32, i32)>>, start: (i32, i32), end: (i32, i32)) {
    edges.entry(start).or_default().push(end);
}

fn take_edge(
    edges: &mut BTreeMap<(i32, i32), Vec<(i32, i32)>>,
    start: (i32, i32),
) -> Option<(i32, i32)> {
    let values = edges.get_mut(&start)?;
    values.sort_unstable();
    let end = values.remove(0);
    if values.is_empty() {
        edges.remove(&start);
    }
    Some(end)
}

/// Simplifies a closed contour with Ramer–Douglas–Peucker approximation.
pub fn approximate_polygon(contour: &Contour, epsilon: f64) -> VisionResult<Contour> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(VisionError::InvalidParameter(
            "polygon epsilon must be finite and non-negative".to_owned(),
        ));
    }
    if contour.points.len() <= 3 || epsilon == 0.0 {
        return Ok(contour.clone());
    }
    // Rotate at the point farthest from vertex 0, then simplify the two open
    // arcs independently so a closed loop has stable endpoints.
    let first = contour.points[0];
    let split = contour
        .points
        .iter()
        .enumerate()
        .max_by_key(|(_, point)| squared_distance(**point, first))
        .map_or(0, |(index, _)| index);
    let mut left = rdp(&contour.points[..=split], epsilon);
    let mut wrapped = contour.points[split..].to_vec();
    wrapped.push(first);
    let right = rdp(&wrapped, epsilon);
    left.extend(right.into_iter().skip(1).take_while(|point| *point != first));
    Ok(Contour { points: left })
}

fn rdp(points: &[[i32; 2]], epsilon: f64) -> Vec<[i32; 2]> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let first = points[0];
    let last = points[points.len() - 1];
    let mut max_distance = 0.0;
    let mut split = 0;
    for (index, &point) in points.iter().enumerate().take(points.len() - 1).skip(1) {
        let distance = point_segment_distance(point, first, last);
        if distance > max_distance {
            max_distance = distance;
            split = index;
        }
    }
    if max_distance <= epsilon {
        return vec![first, last];
    }
    let mut left = rdp(&points[..=split], epsilon);
    let right = rdp(&points[split..], epsilon);
    left.extend(right.into_iter().skip(1));
    left
}

fn squared_distance(a: [i32; 2], b: [i32; 2]) -> i64 {
    let dx = i64::from(a[0]) - i64::from(b[0]);
    let dy = i64::from(a[1]) - i64::from(b[1]);
    dx * dx + dy * dy
}

fn point_segment_distance(point: [i32; 2], start: [i32; 2], end: [i32; 2]) -> f64 {
    let px = f64::from(point[0]);
    let py = f64::from(point[1]);
    let sx = f64::from(start[0]);
    let sy = f64::from(start[1]);
    let dx = f64::from(end[0] - start[0]);
    let dy = f64::from(end[1] - start[1]);
    let length_squared = dx.mul_add(dx, dy * dy);
    if length_squared == 0.0 {
        return (px - sx).hypot(py - sy);
    }
    let t = ((px - sx).mul_add(dx, (py - sy) * dy) / length_squared).clamp(0.0, 1.0);
    (px - sx - t * dx).hypot(py - sy - t * dy)
}

/// Linearization order used by run-length encoded masks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RleOrder {
    /// Conventional row-major traversal.
    #[default]
    RowMajor,
    /// COCO-compatible column-major traversal.
    CocoColumnMajor,
}

/// Alternating zero/one run lengths, always beginning with a zero run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskRle {
    /// Mask width.
    pub width: usize,
    /// Mask height.
    pub height: usize,
    /// Linearization order.
    pub order: RleOrder,
    /// Alternating zero and one run lengths.
    pub counts: Vec<usize>,
}

/// Encodes a binary mask into alternating runs.
#[must_use]
pub fn encode_rle(mask: &BinaryMask, order: RleOrder) -> MaskRle {
    let mut counts = vec![0_usize];
    let mut current = 0_u8;
    for index in 0..mask.width() * mask.height() {
        let (x, y) = linear_coordinate(index, mask.width(), mask.height(), order);
        let value = u8::from(mask.contains(x, y));
        if value == current {
            *counts.last_mut().expect("initial run exists") += 1;
        } else {
            counts.push(1);
            current = value;
        }
    }
    MaskRle { width: mask.width(), height: mask.height(), order, counts }
}

/// Decodes a run-length mask and validates its total length.
pub fn decode_rle(rle: &MaskRle) -> VisionResult<BinaryMask> {
    let total = rle
        .width
        .checked_mul(rle.height)
        .ok_or_else(|| VisionError::InvalidDimensions("RLE dimensions overflow".to_owned()))?;
    let encoded = rle.counts.iter().try_fold(0_usize, |sum, &count| sum.checked_add(count));
    if encoded != Some(total) {
        return Err(VisionError::InvalidParameter(
            "RLE run lengths do not match mask dimensions".to_owned(),
        ));
    }
    let mut data = vec![0_u8; total];
    let mut linear = 0;
    let mut value = 0_u8;
    for &count in &rle.counts {
        for _ in 0..count {
            let (x, y) = linear_coordinate(linear, rle.width, rle.height, rle.order);
            data[y * rle.width + x] = value;
            linear += 1;
        }
        value ^= 1;
    }
    BinaryMask::try_new(rle.width, rle.height, data)
}

fn linear_coordinate(index: usize, width: usize, height: usize, order: RleOrder) -> (usize, usize) {
    match order {
        RleOrder::RowMajor => (index % width, index / width),
        RleOrder::CocoColumnMajor => (index / height, index % height),
    }
}

/// Metric or relative dense depth map. Non-positive and non-finite values are invalid.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthMap {
    image: Image<f32, 1>,
}

impl DepthMap {
    /// Wraps a depth image and marks its semantic metadata.
    pub fn try_new(width: usize, height: usize, data: Vec<f32>) -> VisionResult<Self> {
        if data.iter().any(|value| value.is_infinite()) {
            return Err(VisionError::InvalidParameter(
                "depth values may be finite or NaN, but not infinite".to_owned(),
            ));
        }
        let metadata = ImageMetadata { color_space: ColorSpace::Depth, ..Default::default() };
        Ok(Self { image: Image::try_new_with_metadata(width, height, data, metadata)? })
    }

    /// Borrows the depth image.
    #[must_use]
    pub fn image(&self) -> &Image<f32, 1> {
        &self.image
    }

    /// Builds a validity mask for an inclusive depth range.
    pub fn valid_mask(&self, min_depth: f32, max_depth: f32) -> VisionResult<BinaryMask> {
        if !min_depth.is_finite() || min_depth < 0.0 || max_depth.is_nan() || max_depth < min_depth
        {
            return Err(VisionError::InvalidParameter("invalid depth range".to_owned()));
        }
        BinaryMask::try_new(
            self.image.width(),
            self.image.height(),
            self.image
                .as_slice()
                .iter()
                .map(|&value| {
                    u8::from(value.is_finite() && value >= min_depth && value <= max_depth)
                })
                .collect(),
        )
    }
}

/// Dense confidence values constrained to `[0, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfidenceMap {
    image: Image<f32, 1>,
}

impl ConfidenceMap {
    /// Creates a validated confidence map.
    pub fn try_new(width: usize, height: usize, data: Vec<f32>) -> VisionResult<Self> {
        if data.iter().any(|value| !value.is_finite() || !(0.0..=1.0).contains(value)) {
            return Err(VisionError::InvalidParameter(
                "confidence values must be finite and in [0, 1]".to_owned(),
            ));
        }
        Ok(Self { image: Image::try_new(width, height, data)? })
    }

    /// Borrows the confidence image.
    #[must_use]
    pub fn image(&self) -> &Image<f32, 1> {
        &self.image
    }

    /// Returns confidence map width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.image.width()
    }

    /// Returns confidence map height.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.image.height()
    }
}

/// Dense `(dx, dy)` optical-flow field in pixel units.
#[derive(Clone, Debug, PartialEq)]
pub struct FlowField {
    image: Image<f32, 2>,
}

impl FlowField {
    /// Creates a flow field. NaN marks invalid flow; infinity is rejected.
    pub fn try_new(width: usize, height: usize, data: Vec<f32>) -> VisionResult<Self> {
        if data.iter().any(|value| value.is_infinite()) {
            return Err(VisionError::InvalidParameter(
                "flow values may be finite or NaN, but not infinite".to_owned(),
            ));
        }
        Ok(Self { image: Image::try_new(width, height, data)? })
    }

    /// Borrows the flow image.
    #[must_use]
    pub fn image(&self) -> &Image<f32, 2> {
        &self.image
    }

    /// Converts displacement vectors to absolute remap coordinates.
    pub fn to_remap(&self) -> VisionResult<(Image<f32, 1>, Image<f32, 1>)> {
        let mut map_x = Vec::with_capacity(self.image.width() * self.image.height());
        let mut map_y = Vec::with_capacity(self.image.width() * self.image.height());
        for y in 0..self.image.height() {
            for x in 0..self.image.width() {
                let flow = self.image.get(x, y).expect("coordinate in bounds");
                map_x.push(x as f32 + flow[0]);
                map_y.push(y as f32 + flow[1]);
            }
        }
        Ok((
            Image::try_new(self.image.width(), self.image.height(), map_x)?,
            Image::try_new(self.image.width(), self.image.height(), map_y)?,
        ))
    }
}

/// Dense per-pixel XYZ point map. Non-finite triples represent invalid points.
#[derive(Clone, Debug, PartialEq)]
pub struct PointMap {
    image: Image<f32, 3>,
}

impl PointMap {
    /// Creates a point map and rejects infinite components.
    pub fn try_new(width: usize, height: usize, data: Vec<f32>) -> VisionResult<Self> {
        if data.iter().any(|value| value.is_infinite()) {
            return Err(VisionError::InvalidParameter(
                "point-map values may be finite or NaN, but not infinite".to_owned(),
            ));
        }
        Ok(Self { image: Image::try_new(width, height, data)? })
    }

    /// Borrows the XYZ image.
    #[must_use]
    pub fn image(&self) -> &Image<f32, 3> {
        &self.image
    }

    /// Returns point-map width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.image.width()
    }

    /// Returns point-map height.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.image.height()
    }

    /// Returns a mask where all XYZ components are finite.
    pub fn valid_mask(&self) -> VisionResult<BinaryMask> {
        let mut data = Vec::with_capacity(self.image.width() * self.image.height());
        for y in 0..self.image.height() {
            for x in 0..self.image.width() {
                let point = self.image.get(x, y).expect("coordinate in bounds");
                data.push(u8::from(point.iter().all(|value| value.is_finite())));
            }
        }
        BinaryMask::try_new(self.image.width(), self.image.height(), data)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        approximate_polygon, connected_components, decode_rle, distance_transform_edt,
        distance_transform_edt_with_spacing, encode_rle, find_contours, BinaryMask, Connectivity,
        DepthMap, FlowField, PointMap, RleOrder,
    };
    use spatialrust_image::Image;

    #[test]
    fn threshold_and_components_find_two_regions() {
        let scores = Image::<f32, 1>::try_new(
            4,
            3,
            vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let mask = BinaryMask::from_threshold(scores.view(), 0.5).unwrap();
        let result = connected_components(&mask, Connectivity::Four).unwrap();
        assert_eq!(result.components.len(), 2);
        assert_eq!(result.components[0].area, 4);
        assert_eq!(result.components[1].area, 2);
        assert_eq!(result.components[0].centroid, [1.0, 1.0]);
    }

    #[test]
    fn contours_trace_outer_and_hole_loops() {
        let mask = BinaryMask::try_new(3, 3, vec![1, 1, 1, 1, 0, 1, 1, 1, 1]).unwrap();
        let contours = find_contours(&mask);
        assert_eq!(contours.len(), 2);
        assert!(contours.iter().all(|contour| contour.points.len() >= 4));
        let simplified = approximate_polygon(&contours[0], 0.1).unwrap();
        assert!(simplified.points.len() <= contours[0].points.len());
    }

    #[test]
    fn rle_roundtrips_both_orders() {
        let mask = BinaryMask::try_new(3, 2, vec![0, 1, 1, 1, 0, 1]).unwrap();
        for order in [RleOrder::RowMajor, RleOrder::CocoColumnMajor] {
            let encoded = encode_rle(&mask, order);
            assert_eq!(decode_rle(&encoded).unwrap(), mask);
        }
    }

    #[test]
    fn exact_distance_transform_matches_known_grid() {
        let mask = BinaryMask::try_new(4, 3, vec![0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).unwrap();
        let distances = distance_transform_edt(&mask).unwrap();
        let expected = [
            0.0,
            1.0,
            2.0,
            3.0,
            1.0,
            2.0_f32.sqrt(),
            5.0_f32.sqrt(),
            10.0_f32.sqrt(),
            2.0,
            5.0_f32.sqrt(),
            8.0_f32.sqrt(),
            13.0_f32.sqrt(),
        ];
        for (&actual, expected) in distances.as_slice().iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn distance_transform_respects_anisotropic_spacing() {
        let mask = BinaryMask::try_new(3, 2, vec![0, 1, 1, 1, 1, 1]).unwrap();
        let distances = distance_transform_edt_with_spacing(&mask, 2.0, 3.0).unwrap();
        let expected = [0.0, 2.0, 4.0, 3.0, 13.0_f32.sqrt(), 5.0];
        for (&actual, expected) in distances.as_slice().iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn distance_transform_defines_empty_and_rejects_missing_background() {
        let empty = BinaryMask::try_new(0, 0, Vec::new()).unwrap();
        assert!(distance_transform_edt(&empty).unwrap().as_slice().is_empty());

        let foreground = BinaryMask::try_new(2, 2, vec![1; 4]).unwrap();
        assert!(distance_transform_edt(&foreground).is_err());
        let background = BinaryMask::try_new(2, 2, vec![0; 4]).unwrap();
        assert_eq!(distance_transform_edt(&background).unwrap().as_slice(), &[0.0; 4]);
        assert!(distance_transform_edt_with_spacing(&background, 0.0, 1.0).is_err());
        assert!(distance_transform_edt_with_spacing(&background, 1.0, f32::NAN).is_err());
    }

    #[test]
    fn depth_flow_and_point_maps_expose_validity() {
        let depth = DepthMap::try_new(2, 1, vec![1.0, f32::NAN]).unwrap();
        assert_eq!(depth.valid_mask(0.1, 10.0).unwrap().image().as_slice(), &[1, 0]);
        let flow = FlowField::try_new(2, 1, vec![1.0, 2.0, -1.0, 0.0]).unwrap();
        let (mx, my) = flow.to_remap().unwrap();
        assert_eq!(mx.as_slice(), &[1.0, 0.0]);
        assert_eq!(my.as_slice(), &[2.0, 0.0]);
        let points = PointMap::try_new(2, 1, vec![0.0, 0.0, 1.0, f32::NAN, 0.0, 1.0]).unwrap();
        assert_eq!(points.valid_mask().unwrap().image().as_slice(), &[1, 0]);
    }
}
