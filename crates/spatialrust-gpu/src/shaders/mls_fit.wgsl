const FIT_TERMS: u32 = 6u;
fn basis_term(term: u32, s: f32, t: f32) -> f32 {
    switch term {case 0u: {return 1.0;} case 1u: {return s;} case 2u: {return t;}
        case 3u: {return s*s;} case 4u: {return s*t;} default: {return t*t;}}
}
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
    let terms=FIT_TERMS;
    if support<terms {write_result(i,plane,1u);return;}
    let system=i*48u;
    for (var k=0u;k<48u;k+=1u) {systems[system+k]=0.0;}
    for (var n=0u;n<stored;n+=1u) {
        let j=neighbors[base+2u+2u*n];if j==i {continue;}
        let w=weight(base,n);let d=(positions[j].xyz-centroid)/scale;
        let su=dot3(d,u);let tv=dot3(d,v);
        let height=dot3(d,normal);
        for (var a=0u;a<terms;a+=1u) {
            systems[system+36u+a]+=w*basis_term(a,su,tv)*height;
            for (var b=0u;b<terms;b+=1u) {systems[system+a*FIT_TERMS+b]+=w*basis_term(a,su,tv)*basis_term(b,su,tv);}
        }
    }
    var largest=0.0;
    for (var a=0u;a<terms;a+=1u) {
        for (var b=0u;b<terms;b+=1u) {largest=max(largest,abs(systems[system+a*FIT_TERMS+b]));}
    }
    let epsilon=largest*1e-6;
    // In-place Cholesky for the symmetric positive-semidefinite Gram matrix.
    // A relative Schur-complement threshold chooses the plane on rank failure.
    for (var col=0u;col<terms;col+=1u) {
        for (var row=col;row<terms;row+=1u) {
            var value=systems[system+row*FIT_TERMS+col];
            for (var k=0u;k<col;k+=1u) {value-=systems[system+row*FIT_TERMS+k]*systems[system+col*FIT_TERMS+k];}
            if row==col {
                if value<=epsilon {write_result(i,plane,1u);return;}
                systems[system+col*FIT_TERMS+col]=sqrt(value);
            } else {systems[system+row*FIT_TERMS+col]=value/systems[system+col*FIT_TERMS+col];}
        }
    }
    for (var row=0u;row<terms;row+=1u) {
        var value=systems[system+36u+row];
        for (var k=0u;k<row;k+=1u) {value-=systems[system+row*FIT_TERMS+k]*systems[system+36u+k];}
        systems[system+36u+row]=value/systems[system+row*FIT_TERMS+row];
    }
    for (var reverse=0u;reverse<terms;reverse+=1u) {
        let row=terms-1u-reverse;var value=systems[system+36u+row];
        for (var k=row+1u;k<terms;k+=1u) {value-=systems[system+k*FIT_TERMS+row]*systems[system+42u+k];}
        systems[system+42u+row]=value/systems[system+row*FIT_TERMS+row];
    }
    let qs=qu/scale;let qt=qv/scale;
    var height=0.0;
    for (var a=0u;a<terms;a+=1u) {height+=systems[system+42u+a]*basis_term(a,qs,qt);}
    write_result(i,plane+normal*(height*scale),2u);
}

@compute @workgroup_size(64)
fn fit_surface(@builtin(global_invocation_id) id: vec3<u32>) {fit_mls_point(id.x);}
