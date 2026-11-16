//! 3-D rotations and rigid transforms: unit [`Quaternion`] (Hamilton convention, `w + xi + yj + zk`), conversion to
//! and from rotation matrices / axis-angle / Euler angles, [`Quaternion::slerp`], rigid [`Se3`] transforms, and
//! [`kabsch`] / [`umeyama`] point-set alignment.

use crate::classical::jacobi_eigh;
use crate::error::{Error, Result};

pub type Vec3 = [f64; 3];
pub type Mat3 = [[f64; 3]; 3];

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

pub fn mat_vec(m: &Mat3, v: Vec3) -> Vec3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

pub fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}

pub fn transpose(m: &Mat3) -> Mat3 {
    [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]
}

pub fn det(m: &Mat3) -> f64 {
    dot(m[0], cross(m[1], m[2]))
}

/// Quaternion `w + xi + yj + zk`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub const IDENTITY: Quaternion = Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(&self) -> Result<Self> {
        let n = self.norm();
        if n < 1e-300 {
            return Err(Error::Msg("Quaternion: cannot normalize the zero quaternion".into()));
        }
        Ok(Self { w: self.w / n, x: self.x / n, y: self.y / n, z: self.z / n })
    }

    pub fn conjugate(&self) -> Self {
        Self { w: self.w, x: -self.x, y: -self.y, z: -self.z }
    }

    /// Rotation by `angle` radians about `axis` (any non-zero length).
    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Result<Self> {
        let n = dot(axis, axis).sqrt();
        if n < 1e-300 {
            return Err(Error::Msg("Quaternion::from_axis_angle: the axis must be non-zero".into()));
        }
        let s = (angle / 2.0).sin() / n;
        Ok(Self { w: (angle / 2.0).cos(), x: axis[0] * s, y: axis[1] * s, z: axis[2] * s })
    }

    /// `(axis, angle)` with angle in `[0, pi]`.
    pub fn to_axis_angle(&self) -> (Vec3, f64) {
        let q = if self.w < 0.0 { Self { w: -self.w, x: -self.x, y: -self.y, z: -self.z } } else { *self };
        let s = (q.x * q.x + q.y * q.y + q.z * q.z).sqrt();
        if s < 1e-12 { ([1.0, 0.0, 0.0], 0.0) } else { ([q.x / s, q.y / s, q.z / s], 2.0 * s.atan2(q.w)) }
    }

    /// Intrinsic Z-Y-X (yaw, pitch, roll) Euler angles in radians.
    pub fn from_euler_zyx(yaw: f64, pitch: f64, roll: f64) -> Self {
        let (cy, sy, cp, sp, cr, sr) = ((yaw / 2.0).cos(), (yaw / 2.0).sin(), (pitch / 2.0).cos(), (pitch / 2.0).sin(), (roll / 2.0).cos(), (roll / 2.0).sin());
        Self { w: cr * cp * cy + sr * sp * sy, x: sr * cp * cy - cr * sp * sy, y: cr * sp * cy + sr * cp * sy, z: cr * cp * sy - sr * sp * cy }
    }

    /// `(yaw, pitch, roll)`; pitch is clamped at the +-90 degree gimbal-lock singularity.
    pub fn to_euler_zyx(&self) -> (f64, f64, f64) {
        let q = self;
        let yaw = (2.0 * (q.w * q.z + q.x * q.y)).atan2(1.0 - 2.0 * (q.y * q.y + q.z * q.z));
        let pitch = (2.0 * (q.w * q.y - q.z * q.x)).clamp(-1.0, 1.0).asin();
        let roll = (2.0 * (q.w * q.x + q.y * q.z)).atan2(1.0 - 2.0 * (q.x * q.x + q.y * q.y));
        (yaw, pitch, roll)
    }

    pub fn to_matrix(&self) -> Mat3 {
        let q = self.normalized().unwrap_or(Self::IDENTITY);
        let (w, x, y, z) = (q.w, q.x, q.y, q.z);
        [[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)], [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)], [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]]
    }

    /// Quaternion of a proper rotation matrix (Shepperd's method for numerical stability).
    pub fn from_matrix(m: &Mat3) -> Result<Self> {
        if (det(m) - 1.0).abs() > 1e-6 || mat_mul(m, &transpose(m)).iter().enumerate().any(|(i, r)| r.iter().enumerate().any(|(j, v)| (v - f64::from(u8::from(i == j))).abs() > 1e-6)) {
            return Err(Error::Msg("Quaternion::from_matrix: the matrix is not a proper rotation".into()));
        }
        let t = m[0][0] + m[1][1] + m[2][2];
        let q = if t > 0.0 {
            let s = (t + 1.0).sqrt() * 2.0;
            Self { w: 0.25 * s, x: (m[2][1] - m[1][2]) / s, y: (m[0][2] - m[2][0]) / s, z: (m[1][0] - m[0][1]) / s }
        } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
            let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
            Self { w: (m[2][1] - m[1][2]) / s, x: 0.25 * s, y: (m[0][1] + m[1][0]) / s, z: (m[0][2] + m[2][0]) / s }
        } else if m[1][1] > m[2][2] {
            let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
            Self { w: (m[0][2] - m[2][0]) / s, x: (m[0][1] + m[1][0]) / s, y: 0.25 * s, z: (m[1][2] + m[2][1]) / s }
        } else {
            let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
            Self { w: (m[1][0] - m[0][1]) / s, x: (m[0][2] + m[2][0]) / s, y: (m[1][2] + m[2][1]) / s, z: 0.25 * s }
        };
        q.normalized()
    }

    /// Rotates a vector.
    pub fn rotate(&self, v: Vec3) -> Vec3 {
        mat_vec(&self.to_matrix(), v)
    }

    /// Spherical linear interpolation along the shortest arc (`t` in `[0, 1]`).
    pub fn slerp(&self, other: &Self, t: f64) -> Self {
        let mut o = *other;
        let mut cos = self.w * o.w + self.x * o.x + self.y * o.y + self.z * o.z;
        if cos < 0.0 {
            o = Self { w: -o.w, x: -o.x, y: -o.y, z: -o.z };
            cos = -cos;
        }
        let (a, b) = if cos > 0.9995 {
            (1.0 - t, t) // nearly parallel: plain lerp, then renormalize
        } else {
            let th = cos.acos();
            (((1.0 - t) * th).sin() / th.sin(), (t * th).sin() / th.sin())
        };
        Self { w: a * self.w + b * o.w, x: a * self.x + b * o.x, y: a * self.y + b * o.y, z: a * self.z + b * o.z }.normalized().unwrap_or(Self::IDENTITY)
    }
}

impl std::ops::Mul for Quaternion {
    type Output = Quaternion;
    /// Hamilton product: `a * b` applies `b` first, then `a`.
    fn mul(self, o: Quaternion) -> Quaternion {
        Quaternion {
            w: self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            x: self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            y: self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            z: self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        }
    }
}

/// Rigid transform `p -> R p + t`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Se3 {
    pub rot: Mat3,
    pub trans: Vec3,
}

impl Se3 {
    pub fn identity() -> Self {
        Self { rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], trans: [0.0; 3] }
    }

    pub fn apply(&self, p: Vec3) -> Vec3 {
        let r = mat_vec(&self.rot, p);
        [r[0] + self.trans[0], r[1] + self.trans[1], r[2] + self.trans[2]]
    }

    /// Composition: `self.compose(other)` applies `other` first.
    pub fn compose(&self, other: &Se3) -> Se3 {
        Se3 { rot: mat_mul(&self.rot, &other.rot), trans: self.apply(other.trans) }
    }

    pub fn inverse(&self) -> Se3 {
        let rt = transpose(&self.rot);
        let t = mat_vec(&rt, self.trans);
        Se3 { rot: rt, trans: [-t[0], -t[1], -t[2]] }
    }

    /// Row-major 4x4 homogeneous matrix.
    pub fn to_matrix4(&self) -> [[f64; 4]; 4] {
        let r = &self.rot;
        [[r[0][0], r[0][1], r[0][2], self.trans[0]], [r[1][0], r[1][1], r[1][2], self.trans[1]], [r[2][0], r[2][1], r[2][2], self.trans[2]], [0.0, 0.0, 0.0, 1.0]]
    }
}

/// Optimal rigid alignment (rotation + translation) of `src` onto `dst` by the Kabsch algorithm; returns the
/// transform and the RMS error after alignment. The rotation is found as the dominant eigenvector of Horn's 4x4
/// matrix, so it is always a proper rotation.
pub fn kabsch(src: &[Vec3], dst: &[Vec3]) -> Result<(Se3, f64)> {
    let (se, _scale, rms) = umeyama_impl(src, dst, false)?;
    Ok((se, rms))
}

/// Similarity alignment (rotation, translation and uniform scale): returns `(transform, scale, rms)` such that
/// `dst ~ scale * R src + t`.
pub fn umeyama(src: &[Vec3], dst: &[Vec3]) -> Result<(Se3, f64, f64)> {
    umeyama_impl(src, dst, true)
}

fn umeyama_impl(src: &[Vec3], dst: &[Vec3], with_scale: bool) -> Result<(Se3, f64, f64)> {
    if src.len() != dst.len() || src.len() < 3 {
        return Err(Error::Msg("alignment: need at least 3 corresponding points".into()));
    }
    let n = src.len() as f64;
    let mean = |p: &[Vec3]| -> Vec3 { [0, 1, 2].map(|k| p.iter().map(|v| v[k]).sum::<f64>() / n) };
    let (ms, md) = (mean(src), mean(dst));
    let s: Vec<Vec3> = src.iter().map(|p| [p[0] - ms[0], p[1] - ms[1], p[2] - ms[2]]).collect();
    let d: Vec<Vec3> = dst.iter().map(|p| [p[0] - md[0], p[1] - md[1], p[2] - md[2]]).collect();
    // cross-covariance M = sum s_i d_i^T; Horn's symmetric 4x4 matrix N has the optimal quaternion as top eigenvector
    let mut m = [[0.0; 3]; 3];
    for (a, b) in s.iter().zip(&d) {
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += a[i] * b[j];
            }
        }
    }
    let (sxx, sxy, sxz, syx, syy, syz, szx, szy, szz) = (m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]);
    let nmat = vec![
        vec![sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        vec![syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        vec![szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        vec![sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let (_, vecs) = jacobi_eigh(&nmat);
    let q = Quaternion { w: vecs[0][0], x: vecs[0][1], y: vecs[0][2], z: vecs[0][3] }.normalized()?;
    let rot = q.to_matrix();
    let scale = if with_scale {
        let num: f64 = s.iter().zip(&d).map(|(a, b)| dot(mat_vec(&rot, *a), *b)).sum();
        let den: f64 = s.iter().map(|a| dot(*a, *a)).sum();
        if den < 1e-300 { 1.0 } else { num / den }
    } else {
        1.0
    };
    let rm = mat_vec(&rot, ms);
    let trans = [md[0] - scale * rm[0], md[1] - scale * rm[1], md[2] - scale * rm[2]];
    let se = Se3 { rot: [[rot[0][0] * scale, rot[0][1] * scale, rot[0][2] * scale], [rot[1][0] * scale, rot[1][1] * scale, rot[1][2] * scale], [rot[2][0] * scale, rot[2][1] * scale, rot[2][2] * scale]], trans };
    let rms = (src.iter().zip(dst).map(|(a, b)| { let p = se.apply(*a); (0..3).map(|k| (p[k] - b[k]).powi(2)).sum::<f64>() }).sum::<f64>() / n).sqrt();
    Ok((Se3 { rot, trans }, scale, rms))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close3(a: Vec3, b: Vec3, tol: f64) {
        assert!((0..3).all(|k| (a[k] - b[k]).abs() < tol), "{a:?} vs {b:?}");
    }

    #[test]
    fn quaternion_rotation_basics() {
        // 90 degrees about z maps x to y
        let q = Quaternion::from_axis_angle([0.0, 0.0, 3.0], std::f64::consts::FRAC_PI_2).unwrap();
        close3(q.rotate([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], 1e-12);
        assert!((q.norm() - 1.0).abs() < 1e-12);
        // composition order: rotate about z then x
        let qx = Quaternion::from_axis_angle([1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_2).unwrap();
        close3((qx * q).rotate([1.0, 0.0, 0.0]), qx.rotate(q.rotate([1.0, 0.0, 0.0])), 1e-12);
        // q * conj(q) = identity
        let id = q * q.conjugate();
        assert!((id.w - 1.0).abs() < 1e-12 && id.x.abs() < 1e-12);
        let (axis, angle) = q.to_axis_angle();
        close3(axis, [0.0, 0.0, 1.0], 1e-12);
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!(Quaternion::from_axis_angle([0.0; 3], 1.0).is_err() && Quaternion { w: 0.0, x: 0.0, y: 0.0, z: 0.0 }.normalized().is_err());
    }

    #[test]
    fn matrix_and_euler_round_trips() {
        for (yaw, pitch, roll) in [(0.3, -0.5, 1.1), (-2.0, 0.4, 0.2), (3.0, 1.2, -2.5)] {
            let q = Quaternion::from_euler_zyx(yaw, pitch, roll);
            let (y2, p2, r2) = q.to_euler_zyx();
            assert!((y2 - yaw).abs() < 1e-12 && (p2 - pitch).abs() < 1e-12 && (r2 - roll).abs() < 1e-12);
            let m = q.to_matrix();
            assert!((det(&m) - 1.0).abs() < 1e-12);
            let q2 = Quaternion::from_matrix(&m).unwrap();
            close3(q2.rotate([0.3, -1.0, 2.0]), q.rotate([0.3, -1.0, 2.0]), 1e-12);
        }
        // 180-degree rotations exercise the non-trace branches
        for axis in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let q = Quaternion::from_axis_angle(axis, std::f64::consts::PI).unwrap();
            let r = Quaternion::from_matrix(&q.to_matrix()).unwrap();
            close3(r.rotate([1.0, 2.0, 3.0]), q.rotate([1.0, 2.0, 3.0]), 1e-12);
        }
        assert!(Quaternion::from_matrix(&[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).is_err());
        // a reflection is rejected
        assert!(Quaternion::from_matrix(&[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).is_err());
    }

    #[test]
    fn slerp_interpolates_the_shortest_arc() {
        let a = Quaternion::IDENTITY;
        let b = Quaternion::from_axis_angle([0.0, 1.0, 0.0], 1.2).unwrap();
        let mid = a.slerp(&b, 0.5);
        let (axis, angle) = mid.to_axis_angle();
        close3(axis, [0.0, 1.0, 0.0], 1e-12);
        assert!((angle - 0.6).abs() < 1e-12);
        let (_, a0) = a.slerp(&b, 0.0).to_axis_angle();
        let (_, a1) = a.slerp(&b, 1.0).to_axis_angle();
        assert!(a0.abs() < 1e-12 && (a1 - 1.2).abs() < 1e-12);
        // -b is the same rotation: slerp must not take the long way round
        let nb = Quaternion { w: -b.w, x: -b.x, y: -b.y, z: -b.z };
        assert!((a.slerp(&nb, 0.5).to_axis_angle().1 - 0.6).abs() < 1e-12);
    }

    #[test]
    fn rigid_transforms_and_alignment() {
        let q = Quaternion::from_euler_zyx(0.4, -0.3, 0.9);
        let t = Se3 { rot: q.to_matrix(), trans: [1.0, -2.0, 0.5] };
        close3(t.inverse().apply(t.apply([0.3, 0.4, 0.5])), [0.3, 0.4, 0.5], 1e-12);
        close3(t.compose(&t.inverse()).apply([5.0, 6.0, 7.0]), [5.0, 6.0, 7.0], 1e-12);
        assert_eq!(t.to_matrix4()[3], [0.0, 0.0, 0.0, 1.0]);
        // Kabsch recovers a known rigid motion exactly
        let src: Vec<Vec3> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 1.0], [-1.0, 0.5, 2.0]];
        let dst: Vec<Vec3> = src.iter().map(|&p| t.apply(p)).collect();
        let (est, rms) = kabsch(&src, &dst).unwrap();
        assert!(rms < 1e-9, "{rms}");
        for k in 0..3 {
            close3(est.rot[k], t.rot[k], 1e-9);
        }
        close3(est.trans, t.trans, 1e-9);
        // a reflected target cannot be matched by a rotation: error stays large (no improper "rotation" returned)
        let mirrored: Vec<Vec3> = src.iter().map(|p| [-p[0], p[1], p[2]]).collect();
        let (m_est, m_rms) = kabsch(&src, &mirrored).unwrap();
        assert!((det(&m_est.rot) - 1.0).abs() < 1e-9 && m_rms > 0.1);
        // Umeyama also recovers a uniform scale
        let scaled: Vec<Vec3> = src.iter().map(|&p| { let r = t.apply(p); [r[0] * 2.5 - 1.0, r[1] * 2.5, r[2] * 2.5] }).collect();
        let (_, s, rms) = umeyama(&src, &scaled).unwrap();
        assert!((s - 2.5).abs() < 1e-9 && rms < 1e-9, "scale {s}, rms {rms}");
        assert!(kabsch(&src[..2], &dst[..2]).is_err() && kabsch(&src, &dst[..4]).is_err());
    }
}
