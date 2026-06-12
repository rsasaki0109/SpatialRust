/// Result of a small dense linear solve.
#[derive(Clone, Debug, PartialEq)]
pub enum LeastSquaresResult {
    /// Unique solution vector.
    Solved(Vec<f64>),
    /// The system is singular or ill-conditioned.
    Singular,
}

/// Solves `A x = b` for square `n x n` systems using Gaussian elimination with partial pivoting.
#[must_use]
pub fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> LeastSquaresResult {
    let n = b.len();
    if n == 0 || a.len() != n || a.iter().any(|row| row.len() != n) {
        return LeastSquaresResult::Singular;
    }

    for col in 0..n {
        let mut pivot_row = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > a[pivot_row][col].abs() {
                pivot_row = row;
            }
        }
        if a[pivot_row][col].abs() < 1e-12 {
            return LeastSquaresResult::Singular;
        }
        if pivot_row != col {
            a.swap(pivot_row, col);
            b.swap(pivot_row, col);
        }

        for row in (col + 1)..n {
            let factor = a[row][col] / a[col][col];
            #[allow(clippy::needless_range_loop)]
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for col in (row + 1)..n {
            sum -= a[row][col] * x[col];
        }
        x[row] = sum / a[row][row];
    }

    LeastSquaresResult::Solved(x)
}

#[cfg(test)]
mod tests {
    use super::{solve_linear_system, LeastSquaresResult};

    #[test]
    fn solves_3x3_system() {
        let a = vec![vec![3.0, 2.0, -1.0], vec![2.0, -2.0, 4.0], vec![-1.0, 0.5, -1.0]];
        let b = vec![1.0, -2.0, 0.0];
        match solve_linear_system(a, b) {
            LeastSquaresResult::Solved(x) => {
                assert!((x[0] - 1.0).abs() < 1e-9);
                assert!((x[1] - -2.0).abs() < 1e-9);
                assert!((x[2] - -2.0).abs() < 1e-9);
            }
            LeastSquaresResult::Singular => panic!("expected unique solution"),
        }
    }
}
