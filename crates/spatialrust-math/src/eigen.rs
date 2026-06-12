use crate::{Mat3, Vec3};

/// Result of a symmetric 3x3 eigendecomposition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SymmetricEigen3 {
    /// Eigenvalues sorted in ascending order.
    pub eigenvalues: [f64; 3],
    /// Eigenvectors stored as columns in row-major matrix form.
    pub eigenvectors: Mat3<f64>,
}

/// Computes eigenvalues and eigenvectors of a symmetric 3x3 matrix.
///
/// Uses Jacobi iterations, suitable for normal estimation and local covariance analysis.
#[must_use]
pub fn symmetric_eigen3(matrix: Mat3<f64>) -> SymmetricEigen3 {
    let mut a = matrix;
    let mut v = Mat3::<f64>::identity();

    for _ in 0..32 {
        let mut p = 0;
        let mut q = 1;
        let mut max = a.m[0][1].abs();
        if a.m[0][2].abs() > max {
            max = a.m[0][2].abs();
            p = 0;
            q = 2;
        }
        if a.m[1][2].abs() > max {
            max = a.m[1][2].abs();
            p = 1;
            q = 2;
        }
        if max < 1e-12 {
            break;
        }

        let app = a.m[p][p];
        let aqq = a.m[q][q];
        let apq = a.m[p][q];
        let phi = 0.5 * (aqq - app) / apq;
        let t = phi.signum() / (phi.abs() + (1.0 + phi * phi).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        let app_new = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        let aqq_new = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a.m[p][p] = app_new;
        a.m[q][q] = aqq_new;
        a.m[p][q] = 0.0;
        a.m[q][p] = 0.0;

        for r in 0..3 {
            if r != p && r != q {
                let arp = a.m[r][p];
                let arq = a.m[r][q];
                a.m[r][p] = c * arp - s * arq;
                a.m[p][r] = a.m[r][p];
                a.m[r][q] = s * arp + c * arq;
                a.m[q][r] = a.m[r][q];
            }
        }

        for r in 0..3 {
            let vr_p = v.m[r][p];
            let vr_q = v.m[r][q];
            v.m[r][p] = c * vr_p - s * vr_q;
            v.m[r][q] = s * vr_p + c * vr_q;
        }
    }

    let eigenvalues = [a.m[0][0], a.m[1][1], a.m[2][2]];
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());

    let sorted_values = [eigenvalues[order[0]], eigenvalues[order[1]], eigenvalues[order[2]]];
    let sorted_vectors = Mat3::from_rows(
        [v.m[0][order[0]], v.m[0][order[1]], v.m[0][order[2]]],
        [v.m[1][order[0]], v.m[1][order[1]], v.m[1][order[2]]],
        [v.m[2][order[0]], v.m[2][order[1]], v.m[2][order[2]]],
    );

    SymmetricEigen3 { eigenvalues: sorted_values, eigenvectors: sorted_vectors }
}

/// Returns the eigenvector for the smallest eigenvalue.
#[must_use]
pub fn smallest_eigenvector(matrix: Mat3<f64>) -> Vec3<f64> {
    let result = symmetric_eigen3(matrix);
    Vec3::new(result.eigenvectors.m[0][0], result.eigenvectors.m[1][0], result.eigenvectors.m[2][0])
        .normalize()
}

#[cfg(test)]
mod tests {
    use super::{smallest_eigenvector, symmetric_eigen3};
    use crate::{tolerance::approx_eq_f64, Mat3};

    #[test]
    fn diagonal_eigenvalues() {
        let m = Mat3::from_rows([1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]);
        let result = symmetric_eigen3(m);
        assert!(approx_eq_f64(result.eigenvalues[0], 1.0, 1e-9));
        assert!(approx_eq_f64(result.eigenvalues[1], 2.0, 1e-9));
        assert!(approx_eq_f64(result.eigenvalues[2], 3.0, 1e-9));
    }

    #[test]
    fn plane_normal_from_covariance() {
        let m = Mat3::from_rows([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.01]);
        let normal = smallest_eigenvector(m);
        assert!(approx_eq_f64(normal.z.abs(), 1.0, 1e-6));
    }
}
