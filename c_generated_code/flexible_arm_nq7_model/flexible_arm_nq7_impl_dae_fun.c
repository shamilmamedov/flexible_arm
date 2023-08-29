/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) flexible_arm_nq7_impl_dae_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_c0 CASADI_PREFIX(c0)
#define casadi_c1 CASADI_PREFIX(c1)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_f1 CASADI_PREFIX(f1)
#define casadi_f2 CASADI_PREFIX(f2)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

static const casadi_int casadi_s0[18] = {14, 1, 0, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
static const casadi_int casadi_s1[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[21] = {17, 1, 0, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

static const casadi_real casadi_c0[4] = {-2.5329999999999998e+01, 0., 0., -2.5329999999999998e+01};
static const casadi_real casadi_c1[4] = {1.1350000000000000e-01, 0., 0., 1.1350000000000000e-01};

/* eval_aba:(i0[7],i1[7],i2[7])->(o0[7]) */
static int casadi_f1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
  a0=arg[2]? arg[2][0] : 0;
  a1=9.9999968293183472e-01;
  a2=arg[0]? arg[0][1] : 0;
  a3=sin(a2);
  a3=(a1*a3);
  a2=cos(a2);
  a1=(a1*a2);
  a2=arg[1]? arg[1][0] : 0;
  a4=(a1*a2);
  a5=1.2812524000000000e-04;
  a6=arg[1]? arg[1][1] : 0;
  a7=7.9632671073326389e-04;
  a8=(a7*a2);
  a8=(a6+a8);
  a9=(a5*a8);
  a10=6.2500000000000000e-02;
  a11=9.8375000000000004e-02;
  a12=(a10*a8);
  a12=(a11*a12);
  a12=(a10*a12);
  a9=(a9+a12);
  a12=(a4*a9);
  a13=1.4858724000000001e-04;
  a14=(a13*a4);
  a15=(a10*a4);
  a15=(a11*a15);
  a15=(a10*a15);
  a14=(a14+a15);
  a15=(a8*a14);
  a12=(a12-a15);
  a15=arg[0]? arg[0][2] : 0;
  a16=cos(a15);
  a17=(a16*a4);
  a15=sin(a15);
  a2=(a3*a2);
  a18=(a15*a2);
  a17=(a17-a18);
  a18=1.0248051699999999e-03;
  a19=arg[1]? arg[1][2] : 0;
  a20=(a19+a8);
  a21=(a18*a20);
  a22=1.2500000000000000e-01;
  a23=1.9675000000000001e-01;
  a24=(a22*a8);
  a25=(a16*a24);
  a26=(a22*a20);
  a26=(a25+a26);
  a26=(a23*a26);
  a27=(a22*a26);
  a21=(a21+a27);
  a27=(a17*a21);
  a28=1.0657291700000000e-03;
  a29=(a28*a17);
  a30=(a22*a4);
  a31=(a22*a17);
  a31=(a30+a31);
  a31=(a23*a31);
  a32=(a22*a31);
  a29=(a29+a32);
  a32=(a20*a29);
  a27=(a27-a32);
  a32=(a30*a26);
  a33=(a25*a31);
  a32=(a32-a33);
  a27=(a27+a32);
  a32=arg[0]? arg[0][3] : 0;
  a33=cos(a32);
  a34=(a33*a17);
  a32=sin(a32);
  a35=(a16*a2);
  a36=(a15*a4);
  a35=(a35+a36);
  a36=(a32*a35);
  a34=(a34-a36);
  a36=arg[1]? arg[1][3] : 0;
  a37=(a36+a20);
  a38=(a5*a37);
  a39=2.5000000000000000e-01;
  a40=(a39*a20);
  a40=(a25+a40);
  a41=(a33*a40);
  a24=(a15*a24);
  a42=(a32*a24);
  a41=(a41-a42);
  a42=(a10*a37);
  a42=(a41+a42);
  a42=(a11*a42);
  a43=(a10*a42);
  a38=(a38+a43);
  a43=(a34*a38);
  a44=(a13*a34);
  a45=(a39*a17);
  a45=(a30+a45);
  a46=(a10*a34);
  a46=(a45+a46);
  a46=(a11*a46);
  a47=(a10*a46);
  a44=(a44+a47);
  a47=(a37*a44);
  a43=(a43-a47);
  a47=(a45*a42);
  a48=(a41*a46);
  a47=(a47-a48);
  a43=(a43+a47);
  a47=arg[0]? arg[0][4] : 0;
  a48=cos(a47);
  a49=(a48*a34);
  a47=sin(a47);
  a50=(a33*a35);
  a51=(a32*a17);
  a50=(a50+a51);
  a51=(a47*a50);
  a49=(a49-a51);
  a51=arg[1]? arg[1][4] : 0;
  a52=(a51+a37);
  a53=(a5*a52);
  a54=(a22*a37);
  a54=(a41+a54);
  a55=(a48*a54);
  a56=(a33*a24);
  a40=(a32*a40);
  a56=(a56+a40);
  a40=(a47*a56);
  a55=(a55-a40);
  a40=(a10*a52);
  a40=(a55+a40);
  a40=(a11*a40);
  a57=(a10*a40);
  a53=(a53+a57);
  a57=(a49*a53);
  a58=(a13*a49);
  a59=(a22*a34);
  a59=(a45+a59);
  a60=(a10*a49);
  a60=(a59+a60);
  a60=(a11*a60);
  a61=(a10*a60);
  a58=(a58+a61);
  a61=(a52*a58);
  a57=(a57-a61);
  a61=(a59*a40);
  a62=(a55*a60);
  a61=(a61-a62);
  a57=(a57+a61);
  a61=arg[0]? arg[0][5] : 0;
  a62=cos(a61);
  a63=(a62*a49);
  a61=sin(a61);
  a64=(a48*a50);
  a65=(a47*a34);
  a64=(a64+a65);
  a65=(a61*a64);
  a63=(a63-a65);
  a65=arg[1]? arg[1][5] : 0;
  a66=(a65+a52);
  a18=(a18*a66);
  a67=(a22*a52);
  a67=(a55+a67);
  a68=(a62*a67);
  a69=(a48*a56);
  a54=(a47*a54);
  a69=(a69+a54);
  a54=(a61*a69);
  a68=(a68-a54);
  a54=(a22*a66);
  a54=(a68+a54);
  a54=(a23*a54);
  a70=(a22*a54);
  a18=(a18+a70);
  a70=(a63*a18);
  a28=(a28*a63);
  a71=(a22*a49);
  a71=(a59+a71);
  a72=(a22*a63);
  a72=(a71+a72);
  a72=(a23*a72);
  a73=(a22*a72);
  a28=(a28+a73);
  a73=(a66*a28);
  a70=(a70-a73);
  a73=(a71*a54);
  a74=(a68*a72);
  a73=(a73-a74);
  a70=(a70+a73);
  a73=arg[0]? arg[0][6] : 0;
  a74=cos(a73);
  a75=(a74*a63);
  a73=sin(a73);
  a76=(a62*a64);
  a77=(a61*a49);
  a76=(a76+a77);
  a77=(a73*a76);
  a75=(a75-a77);
  a77=arg[1]? arg[1][6] : 0;
  a78=(a77+a66);
  a5=(a5*a78);
  a79=(a39*a66);
  a79=(a68+a79);
  a80=(a74*a79);
  a81=(a62*a69);
  a67=(a61*a67);
  a81=(a81+a67);
  a67=(a73*a81);
  a80=(a80-a67);
  a67=(a10*a78);
  a67=(a80+a67);
  a67=(a11*a67);
  a82=(a10*a67);
  a5=(a5+a82);
  a82=(a75*a5);
  a13=(a13*a75);
  a83=(a39*a63);
  a83=(a71+a83);
  a84=(a10*a75);
  a84=(a83+a84);
  a84=(a11*a84);
  a10=(a10*a84);
  a13=(a13+a10);
  a10=(a78*a13);
  a82=(a82-a10);
  a10=(a83*a67);
  a85=(a80*a84);
  a10=(a10-a85);
  a82=(a82+a10);
  a10=2.0527583300000005e-05;
  a85=(a77*a75);
  a85=(a10*a85);
  a82=(a82+a85);
  a85=(a74*a82);
  a86=(a74*a76);
  a87=(a73*a63);
  a86=(a86+a87);
  a87=(a10*a86);
  a88=(a78*a87);
  a5=(a86*a5);
  a88=(a88-a5);
  a5=(a74*a81);
  a79=(a73*a79);
  a5=(a5+a79);
  a79=(a5*a84);
  a89=(a11*a5);
  a83=(a83*a89);
  a79=(a79-a83);
  a88=(a88+a79);
  a79=5.3286458375000002e-04;
  a83=(a77*a86);
  a90=(a79*a83);
  a88=(a88-a90);
  a90=(a73*a88);
  a85=(a85-a90);
  a70=(a70+a85);
  a85=4.1055166700000069e-05;
  a90=(a10*a74);
  a91=(a90*a74);
  a92=(a79*a73);
  a93=(a92*a73);
  a91=(a91+a93);
  a91=(a85+a91);
  a93=(a65*a63);
  a94=(a91*a93);
  a90=(a90*a73);
  a92=(a92*a74);
  a90=(a90-a92);
  a92=-6.1484375000000003e-03;
  a95=(a92*a73);
  a96=(a39*a95);
  a90=(a90+a96);
  a96=(a65*a76);
  a97=(a90*a96);
  a94=(a94-a97);
  a70=(a70+a94);
  a94=(a62*a70);
  a97=(a85*a76);
  a98=(a66*a97);
  a18=(a76*a18);
  a98=(a98-a18);
  a18=(a81*a72);
  a99=(a23*a81);
  a71=(a71*a99);
  a18=(a18-a71);
  a98=(a98+a18);
  a82=(a73*a82);
  a88=(a74*a88);
  a82=(a82+a88);
  a88=(a86*a67);
  a18=(a75*a89);
  a88=(a88-a18);
  a83=(a92*a83);
  a88=(a88-a83);
  a83=(a39*a88);
  a82=(a82-a83);
  a98=(a98+a82);
  a82=(a10*a73);
  a83=(a82*a74);
  a18=(a79*a74);
  a71=(a18*a73);
  a83=(a83-a71);
  a71=(a39*a95);
  a83=(a83+a71);
  a71=(a83*a93);
  a100=4.1399479200000004e-03;
  a82=(a82*a73);
  a18=(a18*a74);
  a82=(a82+a18);
  a18=(a92*a74);
  a101=(a39*a18);
  a82=(a82-a101);
  a101=-2.4593750000000001e-02;
  a18=(a101+a18);
  a102=(a39*a18);
  a82=(a82-a102);
  a82=(a100+a82);
  a102=(a82*a96);
  a71=(a71-a102);
  a98=(a98+a71);
  a71=(a61*a98);
  a94=(a94-a71);
  a57=(a57+a94);
  a94=(a62*a91);
  a71=(a61*a83);
  a94=(a94-a71);
  a71=(a94*a62);
  a102=(a62*a90);
  a103=(a61*a82);
  a102=(a102-a103);
  a103=(a102*a61);
  a71=(a71-a103);
  a71=(a10+a71);
  a103=(a51*a49);
  a104=(a71*a103);
  a94=(a94*a61);
  a102=(a102*a62);
  a94=(a94+a102);
  a102=(a95*a62);
  a18=(a101+a18);
  a105=(a18*a61);
  a102=(a102+a105);
  a105=(a22*a102);
  a94=(a94+a105);
  a105=(a51*a64);
  a106=(a94*a105);
  a104=(a104-a106);
  a57=(a57+a104);
  a104=(a48*a57);
  a106=(a10*a64);
  a107=(a52*a106);
  a53=(a64*a53);
  a107=(a107-a53);
  a53=(a69*a60);
  a108=(a11*a69);
  a59=(a59*a108);
  a53=(a53-a59);
  a107=(a107+a53);
  a70=(a61*a70);
  a98=(a62*a98);
  a70=(a70+a98);
  a98=(a76*a54);
  a53=(a63*a99);
  a98=(a98-a53);
  a98=(a98+a88);
  a93=(a95*a93);
  a96=(a18*a96);
  a93=(a93+a96);
  a98=(a98-a93);
  a93=(a22*a98);
  a70=(a70-a93);
  a107=(a107+a70);
  a91=(a61*a91);
  a83=(a62*a83);
  a91=(a91+a83);
  a83=(a91*a62);
  a90=(a61*a90);
  a82=(a62*a82);
  a90=(a90+a82);
  a82=(a90*a61);
  a83=(a83-a82);
  a82=(a22*a102);
  a83=(a83+a82);
  a82=(a83*a103);
  a91=(a91*a61);
  a90=(a90*a62);
  a91=(a91+a90);
  a18=(a18*a62);
  a95=(a95*a61);
  a18=(a18-a95);
  a95=(a22*a18);
  a91=(a91-a95);
  a95=-3.6890625000000003e-02;
  a95=(a95+a18);
  a18=(a22*a95);
  a91=(a91-a18);
  a91=(a79+a91);
  a18=(a91*a105);
  a82=(a82-a18);
  a107=(a107+a82);
  a82=(a47*a107);
  a104=(a104-a82);
  a43=(a43+a104);
  a104=(a48*a71);
  a82=(a47*a83);
  a104=(a104-a82);
  a82=(a104*a48);
  a18=(a48*a94);
  a90=(a47*a91);
  a18=(a18-a90);
  a90=(a18*a47);
  a82=(a82-a90);
  a82=(a10+a82);
  a90=(a36*a34);
  a70=(a82*a90);
  a104=(a104*a47);
  a18=(a18*a48);
  a104=(a104+a18);
  a18=(a102*a48);
  a95=(a92+a95);
  a93=(a95*a47);
  a18=(a18+a93);
  a93=(a22*a18);
  a104=(a104+a93);
  a93=(a36*a50);
  a96=(a104*a93);
  a70=(a70-a96);
  a43=(a43+a70);
  a70=(a33*a43);
  a96=(a10*a50);
  a88=(a37*a96);
  a38=(a50*a38);
  a88=(a88-a38);
  a38=(a56*a46);
  a53=(a11*a56);
  a45=(a45*a53);
  a38=(a38-a45);
  a88=(a88+a38);
  a57=(a47*a57);
  a107=(a48*a107);
  a57=(a57+a107);
  a107=(a64*a40);
  a38=(a49*a108);
  a107=(a107-a38);
  a107=(a107+a98);
  a103=(a102*a103);
  a105=(a95*a105);
  a103=(a103+a105);
  a107=(a107-a103);
  a103=(a22*a107);
  a57=(a57-a103);
  a88=(a88+a57);
  a71=(a47*a71);
  a83=(a48*a83);
  a71=(a71+a83);
  a83=(a71*a48);
  a94=(a47*a94);
  a91=(a48*a91);
  a94=(a94+a91);
  a91=(a94*a47);
  a83=(a83-a91);
  a91=(a22*a18);
  a83=(a83+a91);
  a91=(a83*a90);
  a71=(a71*a47);
  a94=(a94*a48);
  a71=(a71+a94);
  a95=(a95*a48);
  a102=(a102*a47);
  a95=(a95-a102);
  a102=(a22*a95);
  a71=(a71-a102);
  a102=-4.9187500000000002e-02;
  a102=(a102+a95);
  a95=(a22*a102);
  a71=(a71-a95);
  a71=(a79+a71);
  a95=(a71*a93);
  a91=(a91-a95);
  a88=(a88+a91);
  a91=(a32*a88);
  a70=(a70-a91);
  a27=(a27+a70);
  a70=(a33*a82);
  a91=(a32*a83);
  a70=(a70-a91);
  a91=(a70*a33);
  a95=(a33*a104);
  a94=(a32*a71);
  a95=(a95-a94);
  a94=(a95*a32);
  a91=(a91-a94);
  a91=(a85+a91);
  a94=(a19*a17);
  a57=(a91*a94);
  a70=(a70*a32);
  a95=(a95*a33);
  a70=(a70+a95);
  a95=(a18*a33);
  a102=(a92+a102);
  a103=(a102*a32);
  a95=(a95+a103);
  a103=(a39*a95);
  a70=(a70+a103);
  a103=(a19*a35);
  a105=(a70*a103);
  a57=(a57-a105);
  a27=(a27+a57);
  a57=(a16*a27);
  a85=(a85*a35);
  a105=(a20*a85);
  a21=(a35*a21);
  a105=(a105-a21);
  a21=(a24*a31);
  a98=(a23*a24);
  a30=(a30*a98);
  a21=(a21-a30);
  a105=(a105+a21);
  a43=(a32*a43);
  a88=(a33*a88);
  a43=(a43+a88);
  a88=(a50*a42);
  a21=(a34*a53);
  a88=(a88-a21);
  a88=(a88+a107);
  a90=(a18*a90);
  a93=(a102*a93);
  a90=(a90+a93);
  a88=(a88-a90);
  a90=(a39*a88);
  a43=(a43-a90);
  a105=(a105+a43);
  a82=(a32*a82);
  a83=(a33*a83);
  a82=(a82+a83);
  a83=(a82*a33);
  a104=(a32*a104);
  a71=(a33*a71);
  a104=(a104+a71);
  a71=(a104*a32);
  a83=(a83-a71);
  a71=(a39*a95);
  a83=(a83+a71);
  a71=(a83*a94);
  a82=(a82*a32);
  a104=(a104*a33);
  a82=(a82+a104);
  a102=(a102*a33);
  a18=(a18*a32);
  a102=(a102-a18);
  a18=(a39*a102);
  a82=(a82-a18);
  a18=-1.2296875000000000e-01;
  a18=(a18+a102);
  a102=(a39*a18);
  a82=(a82-a102);
  a100=(a100+a82);
  a82=(a100*a103);
  a71=(a71-a82);
  a105=(a105+a71);
  a71=(a15*a105);
  a57=(a57-a71);
  a12=(a12+a57);
  a57=(a16*a91);
  a71=(a15*a83);
  a57=(a57-a71);
  a71=(a57*a16);
  a82=(a16*a70);
  a102=(a15*a100);
  a82=(a82-a102);
  a102=(a82*a15);
  a71=(a71-a102);
  a71=(a10+a71);
  a102=(a6*a4);
  a104=(a71*a102);
  a57=(a57*a15);
  a82=(a82*a16);
  a57=(a57+a82);
  a82=(a95*a16);
  a101=(a101+a18);
  a18=(a101*a15);
  a82=(a82+a18);
  a18=(a22*a82);
  a57=(a57+a18);
  a6=(a6*a2);
  a18=(a57*a6);
  a104=(a104-a18);
  a12=(a12+a104);
  a12=(a3*a12);
  a10=(a10*a2);
  a8=(a8*a10);
  a9=(a2*a9);
  a8=(a8-a9);
  a27=(a15*a27);
  a105=(a16*a105);
  a27=(a27+a105);
  a105=(a35*a26);
  a9=(a17*a98);
  a105=(a105-a9);
  a105=(a105+a88);
  a94=(a95*a94);
  a103=(a101*a103);
  a94=(a94+a103);
  a105=(a105-a94);
  a105=(a22*a105);
  a27=(a27-a105);
  a8=(a8+a27);
  a91=(a15*a91);
  a83=(a16*a83);
  a91=(a91+a83);
  a83=(a91*a16);
  a70=(a15*a70);
  a100=(a16*a100);
  a70=(a70+a100);
  a100=(a70*a15);
  a83=(a83-a100);
  a100=(a22*a82);
  a83=(a83+a100);
  a102=(a83*a102);
  a91=(a91*a15);
  a70=(a70*a16);
  a91=(a91+a70);
  a101=(a101*a16);
  a95=(a95*a15);
  a101=(a101-a95);
  a95=(a22*a101);
  a91=(a91-a95);
  a95=-8.6078125000000005e-02;
  a95=(a95+a101);
  a101=(a22*a95);
  a91=(a91-a101);
  a79=(a79+a91);
  a6=(a79*a6);
  a102=(a102-a6);
  a8=(a8+a102);
  a8=(a1*a8);
  a102=arg[2]? arg[2][1] : 0;
  a6=(a7*a102);
  a8=(a8+a6);
  a12=(a12+a8);
  a0=(a0-a12);
  a12=1.1250000000000000e-02;
  a71=(a3*a71);
  a83=(a1*a83);
  a71=(a71+a83);
  a71=(a71*a3);
  a57=(a3*a57);
  a79=(a1*a79);
  a57=(a57+a79);
  a57=(a57*a1);
  a71=(a71+a57);
  a12=(a12+a71);
  a0=(a0/a12);
  a71=9.8100000000000005e+00;
  a92=(a92+a95);
  a92=(a7*a92);
  a92=(a92*a1);
  a82=(a7*a82);
  a82=(a82*a3);
  a92=(a92-a82);
  a92=(a92/a12);
  a92=(a71*a92);
  a0=(a0-a92);
  if (res[0]!=0) res[0][0]=a0;
  a2=(a2*a14);
  a4=(a4*a10);
  a2=(a2-a4);
  a4=arg[2]? arg[2][2] : 0;
  a10=(a77*a80);
  a10=(a11*a10);
  a14=(a75*a84);
  a92=(a78*a67);
  a14=(a14+a92);
  a10=(a10-a14);
  a14=(a74*a10);
  a78=(a78*a89);
  a84=(a86*a84);
  a78=(a78+a84);
  a84=1.1999232039391526e+01;
  a92=arg[2]? arg[2][6] : 0;
  a86=(a86*a13);
  a75=(a75*a87);
  a86=(a86-a75);
  a67=(a5*a67);
  a80=(a80*a89);
  a67=(a67-a80);
  a86=(a86+a67);
  a86=(a92-a86);
  a67=(a84*a86);
  a80=2.4598471757803664e-02;
  a77=(a77*a5);
  a5=(a80*a77);
  a67=(a67-a5);
  a78=(a78+a67);
  a67=(a73*a78);
  a14=(a14-a67);
  a67=(a63*a72);
  a5=(a66*a54);
  a67=(a67+a5);
  a14=(a14-a67);
  a67=(a11*a74);
  a5=(a67*a74);
  a89=(a80*a73);
  a75=(a89*a73);
  a5=(a5+a75);
  a5=(a23+a5);
  a75=(a11*a73);
  a87=(a75*a74);
  a80=(a80*a74);
  a13=(a80*a73);
  a87=(a87-a13);
  a13=(a39*a87);
  a12=4.0990239200000000e-03;
  a75=(a75*a73);
  a80=(a80*a74);
  a75=(a75+a80);
  a80=(a39*a75);
  a82=(a39*a80);
  a82=(a12+a82);
  a95=(a13/a82);
  a57=(a95*a13);
  a5=(a5-a57);
  a57=(a65*a68);
  a79=(a5*a57);
  a67=(a67*a73);
  a89=(a89*a74);
  a67=(a67-a89);
  a89=2.4593750000000001e-02;
  a80=(a89+a80);
  a83=(a95*a80);
  a67=(a67-a83);
  a65=(a65*a81);
  a83=(a67*a65);
  a79=(a79-a83);
  a83=arg[2]? arg[2][5] : 0;
  a28=(a76*a28);
  a63=(a63*a97);
  a28=(a28-a63);
  a81=(a81*a54);
  a68=(a68*a99);
  a81=(a81-a68);
  a28=(a28+a81);
  a10=(a73*a10);
  a78=(a74*a78);
  a10=(a10+a78);
  a78=(a39*a10);
  a92=(a92+a78);
  a28=(a28+a92);
  a28=(a83-a28);
  a92=(a95*a28);
  a79=(a79+a92);
  a14=(a14+a79);
  a79=(a62*a14);
  a66=(a66*a99);
  a76=(a76*a72);
  a66=(a66+a76);
  a66=(a66+a10);
  a10=(a80/a82);
  a13=(a10*a13);
  a87=(a87-a13);
  a13=(a87*a57);
  a75=(a23+a75);
  a80=(a10*a80);
  a75=(a75-a80);
  a80=(a75*a65);
  a13=(a13-a80);
  a80=(a10*a28);
  a13=(a13+a80);
  a66=(a66+a13);
  a13=(a61*a66);
  a79=(a79-a13);
  a13=(a49*a60);
  a80=(a52*a40);
  a13=(a13+a80);
  a79=(a79-a13);
  a13=(a62*a5);
  a80=(a61*a87);
  a13=(a13-a80);
  a80=(a13*a62);
  a76=(a62*a67);
  a72=(a61*a75);
  a76=(a76-a72);
  a72=(a76*a61);
  a80=(a80-a72);
  a80=(a11+a80);
  a5=(a61*a5);
  a87=(a62*a87);
  a5=(a5+a87);
  a87=(a5*a62);
  a67=(a61*a67);
  a75=(a62*a75);
  a67=(a67+a75);
  a75=(a67*a61);
  a87=(a87-a75);
  a75=(a22*a87);
  a72=5.1240258375000007e-04;
  a5=(a5*a61);
  a67=(a67*a62);
  a5=(a5+a67);
  a67=(a22*a5);
  a99=(a22*a67);
  a99=(a72+a99);
  a92=(a75/a99);
  a78=(a92*a75);
  a80=(a80-a78);
  a78=(a51*a55);
  a81=(a80*a78);
  a13=(a13*a61);
  a76=(a76*a62);
  a13=(a13+a76);
  a76=6.1484375000000003e-03;
  a67=(a76+a67);
  a68=(a92*a67);
  a13=(a13-a68);
  a51=(a51*a69);
  a68=(a13*a51);
  a81=(a81-a68);
  a68=arg[2]? arg[2][4] : 0;
  a58=(a64*a58);
  a49=(a49*a106);
  a58=(a58-a49);
  a69=(a69*a40);
  a55=(a55*a108);
  a69=(a69-a55);
  a58=(a58+a69);
  a14=(a61*a14);
  a66=(a62*a66);
  a14=(a14+a66);
  a66=(a22*a14);
  a83=(a83+a66);
  a58=(a58+a83);
  a58=(a68-a58);
  a83=(a92*a58);
  a81=(a81+a83);
  a79=(a79+a81);
  a81=(a48*a79);
  a52=(a52*a108);
  a64=(a64*a60);
  a52=(a52+a64);
  a52=(a52+a14);
  a14=(a67/a99);
  a75=(a14*a75);
  a87=(a87-a75);
  a75=(a87*a78);
  a5=(a11+a5);
  a67=(a14*a67);
  a5=(a5-a67);
  a67=(a5*a51);
  a75=(a75-a67);
  a67=(a14*a58);
  a75=(a75+a67);
  a52=(a52+a75);
  a75=(a47*a52);
  a81=(a81-a75);
  a75=(a34*a46);
  a67=(a37*a42);
  a75=(a75+a67);
  a81=(a81-a75);
  a75=(a48*a80);
  a67=(a47*a87);
  a75=(a75-a67);
  a67=(a75*a48);
  a64=(a48*a13);
  a60=(a47*a5);
  a64=(a64-a60);
  a60=(a64*a47);
  a67=(a67-a60);
  a67=(a11+a67);
  a80=(a47*a80);
  a87=(a48*a87);
  a80=(a80+a87);
  a87=(a80*a48);
  a13=(a47*a13);
  a5=(a48*a5);
  a13=(a13+a5);
  a5=(a13*a47);
  a87=(a87-a5);
  a5=(a22*a87);
  a80=(a80*a47);
  a13=(a13*a48);
  a80=(a80+a13);
  a13=(a22*a80);
  a60=(a22*a13);
  a60=(a72+a60);
  a108=(a5/a60);
  a83=(a108*a5);
  a67=(a67-a83);
  a83=(a36*a41);
  a66=(a67*a83);
  a75=(a75*a47);
  a64=(a64*a48);
  a75=(a75+a64);
  a13=(a76+a13);
  a64=(a108*a13);
  a75=(a75-a64);
  a36=(a36*a56);
  a64=(a75*a36);
  a66=(a66-a64);
  a64=arg[2]? arg[2][3] : 0;
  a44=(a50*a44);
  a34=(a34*a96);
  a44=(a44-a34);
  a56=(a56*a42);
  a41=(a41*a53);
  a56=(a56-a41);
  a44=(a44+a56);
  a79=(a47*a79);
  a52=(a48*a52);
  a79=(a79+a52);
  a52=(a22*a79);
  a68=(a68+a52);
  a44=(a44+a68);
  a44=(a64-a44);
  a68=(a108*a44);
  a66=(a66+a68);
  a81=(a81+a66);
  a66=(a33*a81);
  a37=(a37*a53);
  a50=(a50*a46);
  a37=(a37+a50);
  a37=(a37+a79);
  a79=(a13/a60);
  a5=(a79*a5);
  a87=(a87-a5);
  a5=(a87*a83);
  a11=(a11+a80);
  a13=(a79*a13);
  a11=(a11-a13);
  a13=(a11*a36);
  a5=(a5-a13);
  a13=(a79*a44);
  a5=(a5+a13);
  a37=(a37+a5);
  a5=(a32*a37);
  a66=(a66-a5);
  a5=(a17*a31);
  a13=(a20*a26);
  a5=(a5+a13);
  a66=(a66-a5);
  a5=(a33*a67);
  a13=(a32*a87);
  a5=(a5-a13);
  a13=(a5*a33);
  a80=(a33*a75);
  a50=(a32*a11);
  a80=(a80-a50);
  a50=(a80*a32);
  a13=(a13-a50);
  a13=(a23+a13);
  a67=(a32*a67);
  a87=(a33*a87);
  a67=(a67+a87);
  a87=(a67*a33);
  a75=(a32*a75);
  a11=(a33*a11);
  a75=(a75+a11);
  a11=(a75*a32);
  a87=(a87-a11);
  a11=(a39*a87);
  a67=(a67*a32);
  a75=(a75*a33);
  a67=(a67+a75);
  a75=(a39*a67);
  a50=(a39*a75);
  a12=(a12+a50);
  a50=(a11/a12);
  a46=(a50*a11);
  a13=(a13-a46);
  a46=(a19*a25);
  a53=(a13*a46);
  a5=(a5*a32);
  a80=(a80*a33);
  a5=(a5+a80);
  a89=(a89+a75);
  a75=(a50*a89);
  a5=(a5-a75);
  a19=(a19*a24);
  a75=(a5*a19);
  a53=(a53-a75);
  a29=(a35*a29);
  a17=(a17*a85);
  a29=(a29-a17);
  a24=(a24*a26);
  a25=(a25*a98);
  a24=(a24-a25);
  a29=(a29+a24);
  a81=(a32*a81);
  a37=(a33*a37);
  a81=(a81+a37);
  a37=(a39*a81);
  a64=(a64+a37);
  a29=(a29+a64);
  a29=(a4-a29);
  a64=(a50*a29);
  a53=(a53+a64);
  a66=(a66+a53);
  a66=(a15*a66);
  a20=(a20*a98);
  a35=(a35*a31);
  a20=(a20+a35);
  a20=(a20+a81);
  a81=(a89/a12);
  a11=(a81*a11);
  a87=(a87-a11);
  a11=(a87*a46);
  a23=(a23+a67);
  a89=(a81*a89);
  a23=(a23-a89);
  a89=(a23*a19);
  a11=(a11-a89);
  a89=(a81*a29);
  a11=(a11+a89);
  a20=(a20+a11);
  a20=(a16*a20);
  a66=(a66+a20);
  a66=(a22*a66);
  a4=(a4+a66);
  a2=(a2+a4);
  a102=(a102-a2);
  a13=(a15*a13);
  a87=(a16*a87);
  a13=(a13+a87);
  a87=(a13*a15);
  a5=(a15*a5);
  a23=(a16*a23);
  a5=(a5+a23);
  a23=(a5*a16);
  a87=(a87+a23);
  a87=(a22*a87);
  a23=(a22*a87);
  a72=(a72+a23);
  a102=(a102/a72);
  a13=(a13*a16);
  a5=(a5*a15);
  a13=(a13-a5);
  a13=(a22*a13);
  a13=(a13/a72);
  a3=(a71*a3);
  a13=(a13*a3);
  a76=(a76+a87);
  a76=(a76/a72);
  a71=(a71*a1);
  a76=(a76*a71);
  a13=(a13+a76);
  a7=(a7*a0);
  a13=(a13+a7);
  a102=(a102-a13);
  if (res[0]!=0) res[0][1]=a102;
  a29=(a29/a12);
  a12=(a16*a3);
  a7=(a7+a102);
  a102=(a22*a7);
  a71=(a71+a102);
  a102=(a15*a71);
  a12=(a12+a102);
  a46=(a46+a12);
  a50=(a50*a46);
  a16=(a16*a71);
  a15=(a15*a3);
  a16=(a16-a15);
  a16=(a16-a19);
  a81=(a81*a16);
  a50=(a50+a81);
  a50=(a50+a7);
  a29=(a29-a50);
  if (res[0]!=0) res[0][2]=a29;
  a44=(a44/a60);
  a60=(a33*a46);
  a7=(a7+a29);
  a29=(a39*a7);
  a16=(a16+a29);
  a29=(a32*a16);
  a60=(a60+a29);
  a83=(a83+a60);
  a108=(a108*a83);
  a33=(a33*a16);
  a32=(a32*a46);
  a33=(a33-a32);
  a33=(a33-a36);
  a79=(a79*a33);
  a108=(a108+a79);
  a108=(a108+a7);
  a44=(a44-a108);
  if (res[0]!=0) res[0][3]=a44;
  a58=(a58/a99);
  a99=(a48*a83);
  a7=(a7+a44);
  a44=(a22*a7);
  a33=(a33+a44);
  a44=(a47*a33);
  a99=(a99+a44);
  a78=(a78+a99);
  a92=(a92*a78);
  a48=(a48*a33);
  a47=(a47*a83);
  a48=(a48-a47);
  a48=(a48-a51);
  a14=(a14*a48);
  a92=(a92+a14);
  a92=(a92+a7);
  a58=(a58-a92);
  if (res[0]!=0) res[0][4]=a58;
  a28=(a28/a82);
  a82=(a62*a78);
  a7=(a7+a58);
  a22=(a22*a7);
  a48=(a48+a22);
  a22=(a61*a48);
  a82=(a82+a22);
  a57=(a57+a82);
  a95=(a95*a57);
  a62=(a62*a48);
  a61=(a61*a78);
  a62=(a62-a61);
  a62=(a62-a65);
  a10=(a10*a62);
  a95=(a95+a10);
  a95=(a95+a7);
  a28=(a28-a95);
  if (res[0]!=0) res[0][5]=a28;
  a95=1.9515904714639330e+03;
  a95=(a95*a86);
  a7=(a7+a28);
  a39=(a39*a7);
  a62=(a62+a39);
  a74=(a74*a62);
  a73=(a73*a57);
  a74=(a74-a73);
  a74=(a74-a77);
  a84=(a84*a74);
  a84=(a84+a7);
  a95=(a95-a84);
  if (res[0]!=0) res[0][6]=a95;
  return 0;
}

/* eval_fkp:(i0[7])->(o0[3]) */
static int casadi_f2(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a3, a4, a5, a6, a7, a8, a9;
  a0=1.2500000000000000e-01;
  a1=arg[0]? arg[0][0] : 0;
  a2=cos(a1);
  a3=arg[0]? arg[0][1] : 0;
  a4=cos(a3);
  a5=(a2*a4);
  a1=sin(a1);
  a6=7.9632671073326389e-04;
  a3=sin(a3);
  a7=(a6*a3);
  a8=(a1*a7);
  a5=(a5-a8);
  a8=(a0*a5);
  a9=2.5000000000000000e-01;
  a10=arg[0]? arg[0][2] : 0;
  a11=cos(a10);
  a12=(a5*a11);
  a13=(a2*a3);
  a6=(a6*a4);
  a14=(a1*a6);
  a13=(a13+a14);
  a10=sin(a10);
  a14=(a13*a10);
  a12=(a12-a14);
  a14=(a9*a12);
  a8=(a8+a14);
  a14=arg[0]? arg[0][3] : 0;
  a15=cos(a14);
  a16=(a12*a15);
  a5=(a5*a10);
  a13=(a13*a11);
  a5=(a5+a13);
  a14=sin(a14);
  a13=(a5*a14);
  a16=(a16-a13);
  a13=(a0*a16);
  a8=(a8+a13);
  a13=arg[0]? arg[0][4] : 0;
  a17=cos(a13);
  a18=(a16*a17);
  a12=(a12*a14);
  a5=(a5*a15);
  a12=(a12+a5);
  a13=sin(a13);
  a5=(a12*a13);
  a18=(a18-a5);
  a5=(a0*a18);
  a8=(a8+a5);
  a5=arg[0]? arg[0][5] : 0;
  a19=cos(a5);
  a20=(a18*a19);
  a16=(a16*a13);
  a12=(a12*a17);
  a16=(a16+a12);
  a5=sin(a5);
  a12=(a16*a5);
  a20=(a20-a12);
  a12=(a9*a20);
  a8=(a8+a12);
  a12=arg[0]? arg[0][6] : 0;
  a21=cos(a12);
  a20=(a20*a21);
  a18=(a18*a5);
  a16=(a16*a19);
  a18=(a18+a16);
  a12=sin(a12);
  a18=(a18*a12);
  a20=(a20-a18);
  a20=(a0*a20);
  a8=(a8+a20);
  if (res[0]!=0) res[0][0]=a8;
  a8=(a1*a4);
  a7=(a2*a7);
  a8=(a8+a7);
  a7=(a0*a8);
  a20=(a8*a11);
  a2=(a2*a6);
  a1=(a1*a3);
  a2=(a2-a1);
  a1=(a2*a10);
  a20=(a20+a1);
  a1=(a9*a20);
  a7=(a7+a1);
  a1=(a20*a15);
  a2=(a2*a11);
  a8=(a8*a10);
  a2=(a2-a8);
  a8=(a2*a14);
  a1=(a1+a8);
  a8=(a0*a1);
  a7=(a7+a8);
  a8=(a1*a17);
  a2=(a2*a15);
  a20=(a20*a14);
  a2=(a2-a20);
  a20=(a2*a13);
  a8=(a8+a20);
  a20=(a0*a8);
  a7=(a7+a20);
  a20=(a8*a19);
  a2=(a2*a17);
  a1=(a1*a13);
  a2=(a2-a1);
  a1=(a2*a5);
  a20=(a20+a1);
  a1=(a9*a20);
  a7=(a7+a1);
  a20=(a20*a21);
  a2=(a2*a19);
  a8=(a8*a5);
  a2=(a2-a8);
  a2=(a2*a12);
  a20=(a20+a2);
  a20=(a0*a20);
  a7=(a7+a20);
  if (res[0]!=0) res[0][1]=a7;
  a7=1.4999999999999999e-01;
  a20=9.9999968293183472e-01;
  a3=(a20*a3);
  a2=(a0*a3);
  a7=(a7+a2);
  a2=(a3*a11);
  a20=(a20*a4);
  a4=(a20*a10);
  a2=(a2+a4);
  a4=(a9*a2);
  a7=(a7+a4);
  a4=(a2*a15);
  a20=(a20*a11);
  a3=(a3*a10);
  a20=(a20-a3);
  a3=(a20*a14);
  a4=(a4+a3);
  a3=(a0*a4);
  a7=(a7+a3);
  a3=(a4*a17);
  a20=(a20*a15);
  a2=(a2*a14);
  a20=(a20-a2);
  a2=(a20*a13);
  a3=(a3+a2);
  a2=(a0*a3);
  a7=(a7+a2);
  a2=(a3*a19);
  a20=(a20*a17);
  a4=(a4*a13);
  a20=(a20-a4);
  a4=(a20*a5);
  a2=(a2+a4);
  a9=(a9*a2);
  a7=(a7+a9);
  a2=(a2*a21);
  a20=(a20*a19);
  a3=(a3*a5);
  a20=(a20-a3);
  a20=(a20*a12);
  a2=(a2+a20);
  a0=(a0*a2);
  a7=(a7+a0);
  if (res[0]!=0) res[0][2]=a7;
  return 0;
}

/* flexible_arm_nq7_impl_dae_fun:(i0[14],i1[14],i2[3],i3[3],i4[])->(o0[17]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real **res1=res+1, *rr, *ss, *tt;
  const casadi_real **arg1=arg+5, *cs;
  casadi_real *w0=w+109, *w1=w+123, *w2=w+130, *w3=w+137, *w4=w+140, *w5=w+142, *w6=w+144, *w7=w+148, *w8=w+150, w9, *w10=w+153, *w11=w+155, *w12=w+162, *w13=w+169, *w14=w+183;
  /* #0: @0 = input[1][0] */
  casadi_copy(arg[1], 14, w0);
  /* #1: @1 = input[0][1] */
  casadi_copy(arg[0] ? arg[0]+7 : 0, 7, w1);
  /* #2: @2 = input[0][0] */
  casadi_copy(arg[0], 7, w2);
  /* #3: @3 = input[2][0] */
  casadi_copy(arg[2], 3, w3);
  /* #4: @4 = @3[:2] */
  for (rr=w4, ss=w3+0; ss!=w3+2; ss+=1) *rr++ = *ss;
  /* #5: @5 = zeros(2x1) */
  casadi_clear(w5, 2);
  /* #6: @6 = 
  [[-25.33, -0], 
   [-0, -25.33]] */
  casadi_copy(casadi_c0, 4, w6);
  /* #7: @7 = @2[2:4] */
  for (rr=w7, ss=w2+2; ss!=w2+4; ss+=1) *rr++ = *ss;
  /* #8: @5 = mac(@6,@7,@5) */
  for (i=0, rr=w5; i<1; ++i) for (j=0; j<2; ++j, ++rr) for (k=0, ss=w6+j, tt=w7+i*2; k<2; ++k) *rr += ss[k*2]**tt++;
  /* #9: @7 = zeros(2x1) */
  casadi_clear(w7, 2);
  /* #10: @6 = 
  [[0.1135, 0], 
   [0, 0.1135]] */
  casadi_copy(casadi_c1, 4, w6);
  /* #11: @8 = @1[2:4] */
  for (rr=w8, ss=w1+2; ss!=w1+4; ss+=1) *rr++ = *ss;
  /* #12: @7 = mac(@6,@8,@7) */
  for (i=0, rr=w7; i<1; ++i) for (j=0; j<2; ++j, ++rr) for (k=0, ss=w6+j, tt=w8+i*2; k<2; ++k) *rr += ss[k*2]**tt++;
  /* #13: @5 = (@5-@7) */
  for (i=0, rr=w5, cs=w7; i<2; ++i) (*rr++) -= (*cs++);
  /* #14: @9 = @3[2] */
  for (rr=(&w9), ss=w3+2; ss!=w3+3; ss+=1) *rr++ = *ss;
  /* #15: @7 = zeros(2x1) */
  casadi_clear(w7, 2);
  /* #16: @6 = 
  [[-25.33, -0], 
   [-0, -25.33]] */
  casadi_copy(casadi_c0, 4, w6);
  /* #17: @8 = @2[5:7] */
  for (rr=w8, ss=w2+5; ss!=w2+7; ss+=1) *rr++ = *ss;
  /* #18: @7 = mac(@6,@8,@7) */
  for (i=0, rr=w7; i<1; ++i) for (j=0; j<2; ++j, ++rr) for (k=0, ss=w6+j, tt=w8+i*2; k<2; ++k) *rr += ss[k*2]**tt++;
  /* #19: @8 = zeros(2x1) */
  casadi_clear(w8, 2);
  /* #20: @6 = 
  [[0.1135, 0], 
   [0, 0.1135]] */
  casadi_copy(casadi_c1, 4, w6);
  /* #21: @10 = @1[5:7] */
  for (rr=w10, ss=w1+5; ss!=w1+7; ss+=1) *rr++ = *ss;
  /* #22: @8 = mac(@6,@10,@8) */
  for (i=0, rr=w8; i<1; ++i) for (j=0; j<2; ++j, ++rr) for (k=0, ss=w6+j, tt=w10+i*2; k<2; ++k) *rr += ss[k*2]**tt++;
  /* #23: @7 = (@7-@8) */
  for (i=0, rr=w7, cs=w8; i<2; ++i) (*rr++) -= (*cs++);
  /* #24: @11 = vertcat(@4, @5, @9, @7) */
  rr=w11;
  for (i=0, cs=w4; i<2; ++i) *rr++ = *cs++;
  for (i=0, cs=w5; i<2; ++i) *rr++ = *cs++;
  *rr++ = w9;
  for (i=0, cs=w7; i<2; ++i) *rr++ = *cs++;
  /* #25: @12 = eval_aba(@2, @1, @11) */
  arg1[0]=w2;
  arg1[1]=w1;
  arg1[2]=w11;
  res1[0]=w12;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #26: @13 = vertcat(@1, @12) */
  rr=w13;
  for (i=0, cs=w1; i<7; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<7; ++i) *rr++ = *cs++;
  /* #27: @0 = (@0-@13) */
  for (i=0, rr=w0, cs=w13; i<14; ++i) (*rr++) -= (*cs++);
  /* #28: output[0][0] = @0 */
  casadi_copy(w0, 14, res[0]);
  /* #29: @3 = input[3][0] */
  casadi_copy(arg[3], 3, w3);
  /* #30: @14 = eval_fkp(@2) */
  arg1[0]=w2;
  res1[0]=w14;
  if (casadi_f2(arg1, res1, iw, w, 0)) return 1;
  /* #31: @3 = (@3-@14) */
  for (i=0, rr=w3, cs=w14; i<3; ++i) (*rr++) -= (*cs++);
  /* #32: output[0][1] = @3 */
  if (res[0]) casadi_copy(w3, 3, res[0]+14);
  return 0;
}

CASADI_SYMBOL_EXPORT int flexible_arm_nq7_impl_dae_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int flexible_arm_nq7_impl_dae_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int flexible_arm_nq7_impl_dae_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void flexible_arm_nq7_impl_dae_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int flexible_arm_nq7_impl_dae_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void flexible_arm_nq7_impl_dae_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void flexible_arm_nq7_impl_dae_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void flexible_arm_nq7_impl_dae_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int flexible_arm_nq7_impl_dae_fun_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int flexible_arm_nq7_impl_dae_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real flexible_arm_nq7_impl_dae_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* flexible_arm_nq7_impl_dae_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* flexible_arm_nq7_impl_dae_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* flexible_arm_nq7_impl_dae_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s1;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* flexible_arm_nq7_impl_dae_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int flexible_arm_nq7_impl_dae_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 9;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 186;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
