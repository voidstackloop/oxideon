//! Cross-validation batch 17: scalar special functions vs scipy.special (tools/crossval/check_batch17.py).
use oxideon_core::special_ext::*;

#[test]
#[ignore]
fn emit() {
    let xs = [1e-8, 1e-4, 0.01, 0.1, 0.3, 0.5, 0.9, 1.0, 1.5, 1.99, 2.0, 2.01, 3.0, 5.0, 8.0, 12.0, 20.0, 26.0, 40.0, 100.0];
    for &x in &xs {
        println!("erfc_{x:e} {:.17e}", erfc(x));
        println!("erfcx_{x:e} {:.17e}", erfcx(x));
        println!("e1_{x:e} {:.17e}", expint_e1(x));
        println!("fresS_{x:e} {:.17e}", fresnel(x).0);
        println!("fresC_{x:e} {:.17e}", fresnel(x).1);
        println!("dawsn_{x:e} {:.17e}", dawsn(x));
        println!("Si_{x:e} {:.17e}", sici(x).0);
        println!("Ci_{x:e} {:.17e}", sici(x).1);
        println!("spence_{x:e} {:.17e}", spence(x));
        println!("lw0_{x:e} {:.17e}", lambertw(x, 0));
    }
    for &x in &[-5.0, -2.0, -1.0, -0.5, -0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 40.0, 100.0] {
        println!("ei_{x:e} {:.17e}", expint_ei(x));
        println!("erfc_{x:e} {:.17e}", erfc(x));
        println!("erfcx_{x:e} {:.17e}", erfcx(x));
        println!("fresS_{x:e} {:.17e}", fresnel(x).0);
        println!("dawsn_{x:e} {:.17e}", dawsn(x));
    }
    for &m in &[-10.0, -1.0, -0.3, 0.0, 0.1, 0.5, 0.9, 0.99, 0.999999] {
        println!("ellipk_{m:e} {:.17e}", ellipk(m));
        println!("ellipe_{m:e} {:.17e}", ellipe(m));
    }
    for &x in &[-0.35, -0.3, -0.2, -0.05, -1e-3] {
        println!("lwm1_{x:e} {:.17e}", lambertw(x, -1));
        println!("lw0_{x:e} {:.17e}", lambertw(x, 0));
    }
    for n in [0, 1, 2, 5, 10, -3] {
        for &x in &[0.1, 1.0, 3.5, 10.0, 25.0, 60.0] {
            println!("jn_{n}_{x:e} {:.17e}", jn(n, x));
            println!("ive_{n}_{x:e} {:.17e}", ive(n, x));
        }
    }
    // relative accuracy of tiny values, large arguments, negative arguments
    for (n, x) in [(10, 0.1), (25, 0.5), (3, 1e-6), (40, 2.0), (0, 30.0), (1, 100.0), (2, 1e4), (5, 1e6), (0, 1e9), (3, 1e12), (7, 80.0), (3, -2.5), (-3, -40.0), (4, -1e5)] {
        println!("jnx_{n}_{x:e} {:.17e}", jn(n, x));
    }
    for (v, x) in [(40.0, 1.0), (100.0, 1.0), (150.0, 5.0), (60.5, 0.3)] {
        println!("kvex_{v:e}_{x:e} {:.17e}", kve(v, x));
    }
    let inf = f64::INFINITY;
    println!("inf_fresS {:.17e}", fresnel(inf).0);
    println!("inf_fresCneg {:.17e}", fresnel(-inf).1);
    println!("inf_Si {:.17e}", sici(inf).0);
    println!("inf_Ci {:.17e}", sici(inf).1);
    println!("inf_lw {:e}", lambertw(inf, 0));
    println!("inf_ellipk {:.17e}", ellipk(-inf));
    for &v in &[0.0, 0.5, 1.0, 2.5, 7.0] {
        for &x in &[0.01, 0.5, 1.0, 5.0, 30.0] {
            println!("kve_{v:e}_{x:e} {:.17e}", kve(v, x));
        }
    }
    for n in [0u32, 1, 2, 3, 6, 10, 25] {
        for &x in &[-0.9, -0.3, 0.2, 0.7, 1.0, 1.7] {
            println!("leg_{n}_{x:e} {:.15e}", legendre(n, x));
            println!("herm_{n}_{x:e} {:.15e}", hermite(n, x));
            println!("herme_{n}_{x:e} {:.15e}", hermite_e(n, x));
            println!("lag_{n}_{x:e} {:.15e}", laguerre(n, x));
            println!("glag_{n}_{x:e} {:.15e}", genlaguerre(n, 1.5, x));
            println!("cheT_{n}_{x:e} {:.15e}", chebyt(n, x));
            println!("cheU_{n}_{x:e} {:.15e}", chebyu(n, x));
            println!("geg_{n}_{x:e} {:.15e}", gegenbauer(n, 0.75, x));
            println!("jac_{n}_{x:e} {:.15e}", jacobi(n, 0.5, 1.5, x));
        }
    }
}
