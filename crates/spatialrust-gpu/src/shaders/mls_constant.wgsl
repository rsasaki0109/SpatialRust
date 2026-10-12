fn fit_mls_point(i: u32) {
    if i>=params.count {return;}
    if atomicLoad(&status.bad_rows)>0u || atomicLoad(&status.numerical)>0u {return;}
    let scale=frames[i].centroid.w;if scale==0.0 {return;}
    let centroid=frames[i].centroid.xyz;let normal=frames[i].normal.xyz;let u=frames[i].u.xyz;let v=frames[i].v.xyz;
    let p=positions[i].xyz;let base=i*(2u+2u*params.capacity);let stored=neighbors[base];
    var support=0u;
    for (var n=0u;n<stored;n+=1u) {if neighbors[base+2u+2u*n]!=i {support+=1u;}}
    let rel=p-centroid;let qu=dot3(rel,u);let qv=dot3(rel,v);
    let plane=centroid+u*qu+v*qv;
    var sum_weight=0.0;var height_sum=0.0;
    for (var n=0u;n<stored;n+=1u) {
        let j=neighbors[base+2u+2u*n];if j==i {continue;}
        let w=weight(base,n);sum_weight+=w;
        height_sum+=w*dot3(positions[j].xyz-centroid,normal);
    }
    write_result(i,plane+normal*(height_sum/sum_weight),2u);
}
@compute @workgroup_size(64)
fn fit_surface(@builtin(global_invocation_id) id: vec3<u32>) {fit_mls_point(id.x);}
