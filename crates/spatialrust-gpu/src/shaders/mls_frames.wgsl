fn smallest_normal(matrix: array<vec3<f32>,3>, epsilon: f32) -> vec3<f32> {
    var a=matrix;
    var v=array<vec3<f32>,3>(vec3<f32>(1.0,0.0,0.0),vec3<f32>(0.0,1.0,0.0),vec3<f32>(0.0,0.0,1.0));
    for (var iteration=0u;iteration<32u;iteration+=1u) {
        var p=0u;var q=1u;var off=abs(a[0][1]);
        if abs(a[0][2])>off {p=0u;q=2u;off=abs(a[0][2]);}
        if abs(a[1][2])>off {p=1u;q=2u;off=abs(a[1][2]);}
        if off==0.0 || off<epsilon {break;}
        let app=a[p][p];let aqq=a[q][q];let apq=a[p][q];let delta=0.5*(aqq-app);
        let scale=max(abs(delta),abs(apq));let d=delta/scale;let h=apq/scale;
        let sign_delta=select(1.0,-1.0,(bitcast<u32>(delta)&0x80000000u)!=0u);
        let t=sign_delta*h/(abs(d)+sqrt(d*d+h*h));let c=1.0/sqrt(1.0+t*t);let s=t*c;
        a[p][p]=c*c*app-2.0*s*c*apq+s*s*aqq;
        a[q][q]=s*s*app+2.0*s*c*apq+c*c*aqq;a[p][q]=0.0;a[q][p]=0.0;
        for (var r=0u;r<3u;r+=1u) {
            if r!=p && r!=q {
                let arp=a[r][p];let arq=a[r][q];a[r][p]=c*arp-s*arq;a[p][r]=a[r][p];
                a[r][q]=s*arp+c*arq;a[q][r]=a[r][q];
            }
            let vrp=v[r][p];let vrq=v[r][q];v[r][p]=c*vrp-s*vrq;v[r][q]=s*vrp+c*vrq;
        }
    }
    var index=0u;
    if a[1][1]<a[index][index] {index=1u;}
    if a[2][2]<a[index][index] {index=2u;}
    return unit(vec3<f32>(v[0][index],v[1][index],v[2][index]));
}

fn prepare_frame(i: u32) {
    if i>=params.count {return;}
    output[i]=vec4<f32>(0.0);
    frames[i]=Frame(vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0));
    let base=i*(2u+2u*params.capacity);let stored=neighbors[base];let total=neighbors[base+1u];
    if stored>params.capacity || total<stored || total>params.count {reject(i,total,true);return;}
    if stored!=total {reject(i,total,false);return;}
    var support=0u;var scale=0.0;let p=positions[i].xyz;
    for (var n=0u;n<stored;n+=1u) {
        let j=neighbors[base+2u+2u*n];let distance=bitcast<f32>(neighbors[base+3u+2u*n]);
        if j>=params.count || !(distance>=0.0 && distance<=params.radius_squared) {reject(i,total,true);return;}
        if j!=i {
            support+=1u;let d=abs(positions[j].xyz-p);scale=max(scale,max(d.x,max(d.y,d.z)));
        }
    }
    if stored<params.min_neighbors || support==0u || scale==0.0 {write_result(i,p,0u);return;}
    let scale_squared=scale*scale;
    if scale_squared<1.1754943508222875e-38 {numerical(i);return;}
    var sum_w=vec2<f32>(0.0);var mean: array<vec2<f32>,3>;
    for (var n=0u;n<stored;n+=1u) {
        let j=neighbors[base+2u+2u*n];if j==i {continue;}
        let w=weight(base,n);sum_w=add(sum_w,w);let d=(positions[j].xyz-p)/scale;
        for (var r=0u;r<3u;r+=1u) {mean[r]=add(mean[r],w*d[r]);}
    }
    let sum_weight=sum_w.x+sum_w.y;
    if sum_weight<1e-12 {write_result(i,p,0u);return;}
    var offset: vec3<f32>;
    for (var r=0u;r<3u;r+=1u) {offset[r]=(mean[r].x+mean[r].y)/sum_weight;}
    let centroid=p+offset*scale;
    var covariance: array<vec2<f32>,6>;
    for (var n=0u;n<stored;n+=1u) {
        let j=neighbors[base+2u+2u*n];if j==i {continue;}
        let w=weight(base,n);let d=(positions[j].xyz-centroid)/scale;
        let products=array<f32,6>(d.x*d.x,d.y*d.y,d.z*d.z,d.x*d.y,d.x*d.z,d.y*d.z);
        for (var b=0u;b<6u;b+=1u) {covariance[b]=add(covariance[b],w*products[b]);}
    }
    var cov: array<f32,6>;
    for (var b=0u;b<6u;b+=1u) {cov[b]=(covariance[b].x+covariance[b].y)/sum_weight;}
    let normal=smallest_normal(array<vec3<f32>,3>(vec3<f32>(cov[0],cov[3],cov[4]),vec3<f32>(cov[3],cov[1],cov[5]),vec3<f32>(cov[4],cov[5],cov[2])),(1e-12/scale_squared)/sum_weight);
    var helper=vec3<f32>(1.0,0.0,0.0);
    if abs(normal.x)>=0.9 {helper=vec3<f32>(0.0,1.0,0.0);}
    let u=unit(cross3(normal,helper));let v=cross3(normal,u);
    frames[i]=Frame(vec4<f32>(centroid,scale),vec4<f32>(normal,0.0),vec4<f32>(u,0.0),vec4<f32>(v,0.0));
}
@compute @workgroup_size(64)
fn prepare_frames(@builtin(global_invocation_id) id: vec3<u32>) {prepare_frame(id.x);}
@compute @workgroup_size(64)
fn finalize(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x;if i>=params.count {return;}
    if atomicLoad(&status.bad_rows)>0u || atomicLoad(&status.numerical)>0u {output[i]=vec4<f32>(0.0);}
}
