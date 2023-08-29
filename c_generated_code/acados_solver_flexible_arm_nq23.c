/*
 * Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
 * Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
 * Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
 * Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

// standard
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific
#include "flexible_arm_nq23_model/flexible_arm_nq23_model.h"





#include "acados_solver_flexible_arm_nq23.h"

#define NX     FLEXIBLE_ARM_NQ23_NX
#define NZ     FLEXIBLE_ARM_NQ23_NZ
#define NU     FLEXIBLE_ARM_NQ23_NU
#define NP     FLEXIBLE_ARM_NQ23_NP
#define NBX    FLEXIBLE_ARM_NQ23_NBX
#define NBX0   FLEXIBLE_ARM_NQ23_NBX0
#define NBU    FLEXIBLE_ARM_NQ23_NBU
#define NSBX   FLEXIBLE_ARM_NQ23_NSBX
#define NSBU   FLEXIBLE_ARM_NQ23_NSBU
#define NSH    FLEXIBLE_ARM_NQ23_NSH
#define NSG    FLEXIBLE_ARM_NQ23_NSG
#define NSPHI  FLEXIBLE_ARM_NQ23_NSPHI
#define NSHN   FLEXIBLE_ARM_NQ23_NSHN
#define NSGN   FLEXIBLE_ARM_NQ23_NSGN
#define NSPHIN FLEXIBLE_ARM_NQ23_NSPHIN
#define NSBXN  FLEXIBLE_ARM_NQ23_NSBXN
#define NS     FLEXIBLE_ARM_NQ23_NS
#define NSN    FLEXIBLE_ARM_NQ23_NSN
#define NG     FLEXIBLE_ARM_NQ23_NG
#define NBXN   FLEXIBLE_ARM_NQ23_NBXN
#define NGN    FLEXIBLE_ARM_NQ23_NGN
#define NY0    FLEXIBLE_ARM_NQ23_NY0
#define NY     FLEXIBLE_ARM_NQ23_NY
#define NYN    FLEXIBLE_ARM_NQ23_NYN
// #define N      FLEXIBLE_ARM_NQ23_N
#define NH     FLEXIBLE_ARM_NQ23_NH
#define NPHI   FLEXIBLE_ARM_NQ23_NPHI
#define NHN    FLEXIBLE_ARM_NQ23_NHN
#define NPHIN  FLEXIBLE_ARM_NQ23_NPHIN
#define NR     FLEXIBLE_ARM_NQ23_NR


// ** solver data **

flexible_arm_nq23_solver_capsule * flexible_arm_nq23_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(flexible_arm_nq23_solver_capsule));
    flexible_arm_nq23_solver_capsule *capsule = (flexible_arm_nq23_solver_capsule *) capsule_mem;

    return capsule;
}


int flexible_arm_nq23_acados_free_capsule(flexible_arm_nq23_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int flexible_arm_nq23_acados_create(flexible_arm_nq23_solver_capsule* capsule)
{
    int N_shooting_intervals = FLEXIBLE_ARM_NQ23_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return flexible_arm_nq23_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int flexible_arm_nq23_acados_update_time_steps(flexible_arm_nq23_solver_capsule* capsule, int N, double* new_time_steps)
{
    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "flexible_arm_nq23_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;
}

/**
 * Internal function for flexible_arm_nq23_acados_create: step 1
 */
void flexible_arm_nq23_acados_create_1_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/
    nlp_solver_plan->nlp_solver = SQP_RTI;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;

    nlp_solver_plan->nlp_cost[0] = LINEAR_LS;
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = LINEAR_LS;

    nlp_solver_plan->nlp_cost[N] = LINEAR_LS;

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = CONTINUOUS_MODEL;
        nlp_solver_plan->sim_solver_plan[i].sim_solver = IRK;
    }

    for (int i = 0; i < N; i++)
    {nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;
}


/**
 * Internal function for flexible_arm_nq23_acados_create: step 2
 */
ocp_nlp_dims* flexible_arm_nq23_acados_create_2_create_and_set_dimensions(flexible_arm_nq23_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 17
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
    }

    // for initial state
    nbx[0]  = NBX0;
    nsbx[0] = 0;
    ns[0] = NS - NSBX;
    nbxe[0] = 46;
    ny[0] = NY0;

    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, 0, "ny", &ny[0]);
    for (int i = 1; i < N; i++)
        ocp_nlp_dims_set_cost(nlp_config, nlp_dims, i, "ny", &ny[i]);

    for (int i = 0; i < N; i++)
    {
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);
    free(intNp1mem);
return nlp_dims;
}


/**
 * Internal function for flexible_arm_nq23_acados_create: step 3
 */
void flexible_arm_nq23_acados_create_3_create_and_set_functions(flexible_arm_nq23_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_param_casadi_create(&capsule->__CAPSULE_FNC__ , 0); \
    }while(false)




    // implicit dae
    capsule->impl_dae_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(impl_dae_fun[i], flexible_arm_nq23_impl_dae_fun);
    }

    capsule->impl_dae_fun_jac_x_xdot_z = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(impl_dae_fun_jac_x_xdot_z[i], flexible_arm_nq23_impl_dae_fun_jac_x_xdot_z);
    }

    capsule->impl_dae_jac_x_xdot_u_z = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(impl_dae_jac_x_xdot_u_z[i], flexible_arm_nq23_impl_dae_jac_x_xdot_u_z);
    }


#undef MAP_CASADI_FNC
}


/**
 * Internal function for flexible_arm_nq23_acados_create: step 4
 */
void flexible_arm_nq23_acados_create_4_set_default_parameters(flexible_arm_nq23_solver_capsule* capsule) {
    // no parameters defined
}


/**
 * Internal function for flexible_arm_nq23_acados_create: step 5
 */
void flexible_arm_nq23_acados_create_5_set_nlp_in(flexible_arm_nq23_solver_capsule* capsule, const int N, double* new_time_steps)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    /************************************************
    *  nlp_in
    ************************************************/
//    ocp_nlp_in * nlp_in = ocp_nlp_in_create(nlp_config, nlp_dims);
//    capsule->nlp_in = nlp_in;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    // set up time_steps
    

    if (new_time_steps) {
        flexible_arm_nq23_acados_update_time_steps(capsule, N, new_time_steps);
    } else {// all time_steps are identical
        double time_step = 0.01;
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &time_step);
        }
    }

    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "impl_dae_fun", &capsule->impl_dae_fun[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i,
                                   "impl_dae_fun_jac_x_xdot_z", &capsule->impl_dae_fun_jac_x_xdot_z[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i,
                                   "impl_dae_jac_x_xdot_u", &capsule->impl_dae_jac_x_xdot_u_z[i]);
    
    }

    /**** Cost ****/
    double* W_0 = calloc(NY0*NY0, sizeof(double));
    // change only the non-zero elements:
    W_0[0+(NY0) * 0] = 0.1;
    W_0[1+(NY0) * 1] = 0.1;
    W_0[2+(NY0) * 2] = 0.1;
    W_0[3+(NY0) * 3] = 0.1;
    W_0[4+(NY0) * 4] = 0.1;
    W_0[5+(NY0) * 5] = 0.1;
    W_0[6+(NY0) * 6] = 0.1;
    W_0[7+(NY0) * 7] = 0.1;
    W_0[8+(NY0) * 8] = 0.1;
    W_0[9+(NY0) * 9] = 0.1;
    W_0[10+(NY0) * 10] = 0.1;
    W_0[11+(NY0) * 11] = 0.1;
    W_0[12+(NY0) * 12] = 0.1;
    W_0[13+(NY0) * 13] = 0.1;
    W_0[14+(NY0) * 14] = 0.1;
    W_0[15+(NY0) * 15] = 0.1;
    W_0[16+(NY0) * 16] = 0.1;
    W_0[17+(NY0) * 17] = 0.1;
    W_0[18+(NY0) * 18] = 0.1;
    W_0[19+(NY0) * 19] = 0.1;
    W_0[20+(NY0) * 20] = 0.1;
    W_0[21+(NY0) * 21] = 0.1;
    W_0[22+(NY0) * 22] = 0.1;
    W_0[23+(NY0) * 23] = 10;
    W_0[24+(NY0) * 24] = 10;
    W_0[25+(NY0) * 25] = 0.001;
    W_0[26+(NY0) * 26] = 0.001;
    W_0[27+(NY0) * 27] = 0.001;
    W_0[28+(NY0) * 28] = 0.001;
    W_0[29+(NY0) * 29] = 0.001;
    W_0[30+(NY0) * 30] = 0.001;
    W_0[31+(NY0) * 31] = 0.001;
    W_0[32+(NY0) * 32] = 0.001;
    W_0[33+(NY0) * 33] = 0.001;
    W_0[34+(NY0) * 34] = 0.001;
    W_0[35+(NY0) * 35] = 10;
    W_0[36+(NY0) * 36] = 0.001;
    W_0[37+(NY0) * 37] = 0.001;
    W_0[38+(NY0) * 38] = 0.001;
    W_0[39+(NY0) * 39] = 0.001;
    W_0[40+(NY0) * 40] = 0.001;
    W_0[41+(NY0) * 41] = 0.001;
    W_0[42+(NY0) * 42] = 0.001;
    W_0[43+(NY0) * 43] = 0.001;
    W_0[44+(NY0) * 44] = 0.001;
    W_0[45+(NY0) * 45] = 0.001;
    W_0[46+(NY0) * 46] = 0.1;
    W_0[47+(NY0) * 47] = 0.1;
    W_0[48+(NY0) * 48] = 0.1;
    W_0[49+(NY0) * 49] = 3000;
    W_0[50+(NY0) * 50] = 3000;
    W_0[51+(NY0) * 51] = 3000;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "W", W_0);
    free(W_0);

    double* yref_0 = calloc(NY0, sizeof(double));
    // change only the non-zero elements:
    yref_0[0] = 1.5707963267948966;
    yref_0[1] = 0.3141592653589793;
    yref_0[2] = -0.028201378200947976;
    yref_0[3] = -0.025423843544886696;
    yref_0[4] = -0.022773678035968668;
    yref_0[5] = -0.020256198193234612;
    yref_0[6] = -0.01787566099064464;
    yref_0[7] = -0.01563544779039782;
    yref_0[8] = -0.013538223698398973;
    yref_0[9] = -0.011586073537338032;
    yref_0[10] = -0.009780616183575582;
    yref_0[11] = -0.008123099353472996;
    yref_0[12] = -0.39269908169872414;
    yref_0[13] = -0.0066305212370094116;
    yref_0[14] = -0.00530467521219187;
    yref_0[15] = -0.004127814134718859;
    yref_0[16] = -0.0030992169376013973;
    yref_0[17] = -0.002218316178472931;
    yref_0[18] = -0.0014846862713757713;
    yref_0[19] = -0.000898027075052084;
    yref_0[20] = -0.00045814416878752634;
    yref_0[21] = -0.0001649269084287587;
    yref_0[22] = -0.00001832512746249281;
    yref_0[49] = 0.00002364370748213062;
    yref_0[50] = 0.970354181448497;
    yref_0[51] = 0.12030904568850526;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "yref", yref_0);
    free(yref_0);
    double* W = calloc(NY*NY, sizeof(double));
    // change only the non-zero elements:
    W[0+(NY) * 0] = 0.1;
    W[1+(NY) * 1] = 0.1;
    W[2+(NY) * 2] = 0.1;
    W[3+(NY) * 3] = 0.1;
    W[4+(NY) * 4] = 0.1;
    W[5+(NY) * 5] = 0.1;
    W[6+(NY) * 6] = 0.1;
    W[7+(NY) * 7] = 0.1;
    W[8+(NY) * 8] = 0.1;
    W[9+(NY) * 9] = 0.1;
    W[10+(NY) * 10] = 0.1;
    W[11+(NY) * 11] = 0.1;
    W[12+(NY) * 12] = 0.1;
    W[13+(NY) * 13] = 0.1;
    W[14+(NY) * 14] = 0.1;
    W[15+(NY) * 15] = 0.1;
    W[16+(NY) * 16] = 0.1;
    W[17+(NY) * 17] = 0.1;
    W[18+(NY) * 18] = 0.1;
    W[19+(NY) * 19] = 0.1;
    W[20+(NY) * 20] = 0.1;
    W[21+(NY) * 21] = 0.1;
    W[22+(NY) * 22] = 0.1;
    W[23+(NY) * 23] = 10;
    W[24+(NY) * 24] = 10;
    W[25+(NY) * 25] = 0.001;
    W[26+(NY) * 26] = 0.001;
    W[27+(NY) * 27] = 0.001;
    W[28+(NY) * 28] = 0.001;
    W[29+(NY) * 29] = 0.001;
    W[30+(NY) * 30] = 0.001;
    W[31+(NY) * 31] = 0.001;
    W[32+(NY) * 32] = 0.001;
    W[33+(NY) * 33] = 0.001;
    W[34+(NY) * 34] = 0.001;
    W[35+(NY) * 35] = 10;
    W[36+(NY) * 36] = 0.001;
    W[37+(NY) * 37] = 0.001;
    W[38+(NY) * 38] = 0.001;
    W[39+(NY) * 39] = 0.001;
    W[40+(NY) * 40] = 0.001;
    W[41+(NY) * 41] = 0.001;
    W[42+(NY) * 42] = 0.001;
    W[43+(NY) * 43] = 0.001;
    W[44+(NY) * 44] = 0.001;
    W[45+(NY) * 45] = 0.001;
    W[46+(NY) * 46] = 0.1;
    W[47+(NY) * 47] = 0.1;
    W[48+(NY) * 48] = 0.1;
    W[49+(NY) * 49] = 3000;
    W[50+(NY) * 50] = 3000;
    W[51+(NY) * 51] = 3000;

    double* yref = calloc(NY, sizeof(double));
    // change only the non-zero elements:
    yref[0] = 1.5707963267948966;
    yref[1] = 0.3141592653589793;
    yref[2] = -0.028201378200947976;
    yref[3] = -0.025423843544886696;
    yref[4] = -0.022773678035968668;
    yref[5] = -0.020256198193234612;
    yref[6] = -0.01787566099064464;
    yref[7] = -0.01563544779039782;
    yref[8] = -0.013538223698398973;
    yref[9] = -0.011586073537338032;
    yref[10] = -0.009780616183575582;
    yref[11] = -0.008123099353472996;
    yref[12] = -0.39269908169872414;
    yref[13] = -0.0066305212370094116;
    yref[14] = -0.00530467521219187;
    yref[15] = -0.004127814134718859;
    yref[16] = -0.0030992169376013973;
    yref[17] = -0.002218316178472931;
    yref[18] = -0.0014846862713757713;
    yref[19] = -0.000898027075052084;
    yref[20] = -0.00045814416878752634;
    yref[21] = -0.0001649269084287587;
    yref[22] = -0.00001832512746249281;
    yref[49] = 0.00002364370748213062;
    yref[50] = 0.970354181448497;
    yref[51] = 0.12030904568850526;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }
    free(W);
    free(yref);
    double* Vx_0 = calloc(NY0*NX, sizeof(double));
    // change only the non-zero elements:
    Vx_0[0+(NY0) * 0] = 1;
    Vx_0[1+(NY0) * 1] = 1;
    Vx_0[2+(NY0) * 2] = 1;
    Vx_0[3+(NY0) * 3] = 1;
    Vx_0[4+(NY0) * 4] = 1;
    Vx_0[5+(NY0) * 5] = 1;
    Vx_0[6+(NY0) * 6] = 1;
    Vx_0[7+(NY0) * 7] = 1;
    Vx_0[8+(NY0) * 8] = 1;
    Vx_0[9+(NY0) * 9] = 1;
    Vx_0[10+(NY0) * 10] = 1;
    Vx_0[11+(NY0) * 11] = 1;
    Vx_0[12+(NY0) * 12] = 1;
    Vx_0[13+(NY0) * 13] = 1;
    Vx_0[14+(NY0) * 14] = 1;
    Vx_0[15+(NY0) * 15] = 1;
    Vx_0[16+(NY0) * 16] = 1;
    Vx_0[17+(NY0) * 17] = 1;
    Vx_0[18+(NY0) * 18] = 1;
    Vx_0[19+(NY0) * 19] = 1;
    Vx_0[20+(NY0) * 20] = 1;
    Vx_0[21+(NY0) * 21] = 1;
    Vx_0[22+(NY0) * 22] = 1;
    Vx_0[23+(NY0) * 23] = 1;
    Vx_0[24+(NY0) * 24] = 1;
    Vx_0[25+(NY0) * 25] = 1;
    Vx_0[26+(NY0) * 26] = 1;
    Vx_0[27+(NY0) * 27] = 1;
    Vx_0[28+(NY0) * 28] = 1;
    Vx_0[29+(NY0) * 29] = 1;
    Vx_0[30+(NY0) * 30] = 1;
    Vx_0[31+(NY0) * 31] = 1;
    Vx_0[32+(NY0) * 32] = 1;
    Vx_0[33+(NY0) * 33] = 1;
    Vx_0[34+(NY0) * 34] = 1;
    Vx_0[35+(NY0) * 35] = 1;
    Vx_0[36+(NY0) * 36] = 1;
    Vx_0[37+(NY0) * 37] = 1;
    Vx_0[38+(NY0) * 38] = 1;
    Vx_0[39+(NY0) * 39] = 1;
    Vx_0[40+(NY0) * 40] = 1;
    Vx_0[41+(NY0) * 41] = 1;
    Vx_0[42+(NY0) * 42] = 1;
    Vx_0[43+(NY0) * 43] = 1;
    Vx_0[44+(NY0) * 44] = 1;
    Vx_0[45+(NY0) * 45] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vx", Vx_0);
    free(Vx_0);
    double* Vu_0 = calloc(NY0*NU, sizeof(double));
    // change only the non-zero elements:
    Vu_0[46+(NY0) * 0] = 1;
    Vu_0[47+(NY0) * 1] = 1;
    Vu_0[48+(NY0) * 2] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vu", Vu_0);
    free(Vu_0);
    double* Vz_0 = calloc(NY0*NZ, sizeof(double));
    // change only the non-zero elements:
    
    Vz_0[49+(NY0) * 0] = 1;
    Vz_0[50+(NY0) * 1] = 1;
    Vz_0[51+(NY0) * 2] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vz", Vz_0);
    free(Vz_0);
    double* Vx = calloc(NY*NX, sizeof(double));
    // change only the non-zero elements:
    Vx[0+(NY) * 0] = 1;
    Vx[1+(NY) * 1] = 1;
    Vx[2+(NY) * 2] = 1;
    Vx[3+(NY) * 3] = 1;
    Vx[4+(NY) * 4] = 1;
    Vx[5+(NY) * 5] = 1;
    Vx[6+(NY) * 6] = 1;
    Vx[7+(NY) * 7] = 1;
    Vx[8+(NY) * 8] = 1;
    Vx[9+(NY) * 9] = 1;
    Vx[10+(NY) * 10] = 1;
    Vx[11+(NY) * 11] = 1;
    Vx[12+(NY) * 12] = 1;
    Vx[13+(NY) * 13] = 1;
    Vx[14+(NY) * 14] = 1;
    Vx[15+(NY) * 15] = 1;
    Vx[16+(NY) * 16] = 1;
    Vx[17+(NY) * 17] = 1;
    Vx[18+(NY) * 18] = 1;
    Vx[19+(NY) * 19] = 1;
    Vx[20+(NY) * 20] = 1;
    Vx[21+(NY) * 21] = 1;
    Vx[22+(NY) * 22] = 1;
    Vx[23+(NY) * 23] = 1;
    Vx[24+(NY) * 24] = 1;
    Vx[25+(NY) * 25] = 1;
    Vx[26+(NY) * 26] = 1;
    Vx[27+(NY) * 27] = 1;
    Vx[28+(NY) * 28] = 1;
    Vx[29+(NY) * 29] = 1;
    Vx[30+(NY) * 30] = 1;
    Vx[31+(NY) * 31] = 1;
    Vx[32+(NY) * 32] = 1;
    Vx[33+(NY) * 33] = 1;
    Vx[34+(NY) * 34] = 1;
    Vx[35+(NY) * 35] = 1;
    Vx[36+(NY) * 36] = 1;
    Vx[37+(NY) * 37] = 1;
    Vx[38+(NY) * 38] = 1;
    Vx[39+(NY) * 39] = 1;
    Vx[40+(NY) * 40] = 1;
    Vx[41+(NY) * 41] = 1;
    Vx[42+(NY) * 42] = 1;
    Vx[43+(NY) * 43] = 1;
    Vx[44+(NY) * 44] = 1;
    Vx[45+(NY) * 45] = 1;
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vx", Vx);
    }
    free(Vx);

    
    double* Vu = calloc(NY*NU, sizeof(double));
    // change only the non-zero elements:
    
    Vu[46+(NY) * 0] = 1;
    Vu[47+(NY) * 1] = 1;
    Vu[48+(NY) * 2] = 1;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vu", Vu);
    }
    free(Vu);
    double* Vz = calloc(NY*NZ, sizeof(double));
    // change only the non-zero elements:
    
    Vz[49+(NY) * 0] = 1;
    Vz[50+(NY) * 1] = 1;
    Vz[51+(NY) * 2] = 1;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vz", Vz);
    }
    free(Vz);
    double* zlumem = calloc(4*NS, sizeof(double));
    double* Zl = zlumem+NS*0;
    double* Zu = zlumem+NS*1;
    double* zl = zlumem+NS*2;
    double* zu = zlumem+NS*3;
    // change only the non-zero elements:
    Zl[0] = 1000;
    Zl[1] = 1000;
    Zl[2] = 1000;
    Zu[0] = 1000;
    Zu[1] = 1000;
    Zu[2] = 1000;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zl", Zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zu", Zu);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zl", zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zu", zu);
    }
    free(zlumem);

    // terminal cost
    double* yref_e = calloc(NYN, sizeof(double));
    // change only the non-zero elements:
    yref_e[0] = 1.5707963267948966;
    yref_e[1] = 0.3141592653589793;
    yref_e[2] = -0.028201378200947976;
    yref_e[3] = -0.025423843544886696;
    yref_e[4] = -0.022773678035968668;
    yref_e[5] = -0.020256198193234612;
    yref_e[6] = -0.01787566099064464;
    yref_e[7] = -0.01563544779039782;
    yref_e[8] = -0.013538223698398973;
    yref_e[9] = -0.011586073537338032;
    yref_e[10] = -0.009780616183575582;
    yref_e[11] = -0.008123099353472996;
    yref_e[12] = -0.39269908169872414;
    yref_e[13] = -0.0066305212370094116;
    yref_e[14] = -0.00530467521219187;
    yref_e[15] = -0.004127814134718859;
    yref_e[16] = -0.0030992169376013973;
    yref_e[17] = -0.002218316178472931;
    yref_e[18] = -0.0014846862713757713;
    yref_e[19] = -0.000898027075052084;
    yref_e[20] = -0.00045814416878752634;
    yref_e[21] = -0.0001649269084287587;
    yref_e[22] = -0.00001832512746249281;
    yref_e[46] = 0.00002364370748213062;
    yref_e[47] = 0.970354181448497;
    yref_e[48] = 0.12030904568850526;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);
    free(yref_e);

    double* W_e = calloc(NYN*NYN, sizeof(double));
    // change only the non-zero elements:
    W_e[0+(NYN) * 0] = 0.1;
    W_e[1+(NYN) * 1] = 0.1;
    W_e[2+(NYN) * 2] = 0.1;
    W_e[3+(NYN) * 3] = 0.1;
    W_e[4+(NYN) * 4] = 0.1;
    W_e[5+(NYN) * 5] = 0.1;
    W_e[6+(NYN) * 6] = 0.1;
    W_e[7+(NYN) * 7] = 0.1;
    W_e[8+(NYN) * 8] = 0.1;
    W_e[9+(NYN) * 9] = 0.1;
    W_e[10+(NYN) * 10] = 0.1;
    W_e[11+(NYN) * 11] = 0.1;
    W_e[12+(NYN) * 12] = 0.1;
    W_e[13+(NYN) * 13] = 0.1;
    W_e[14+(NYN) * 14] = 0.1;
    W_e[15+(NYN) * 15] = 0.1;
    W_e[16+(NYN) * 16] = 0.1;
    W_e[17+(NYN) * 17] = 0.1;
    W_e[18+(NYN) * 18] = 0.1;
    W_e[19+(NYN) * 19] = 0.1;
    W_e[20+(NYN) * 20] = 0.1;
    W_e[21+(NYN) * 21] = 0.1;
    W_e[22+(NYN) * 22] = 0.1;
    W_e[23+(NYN) * 23] = 1;
    W_e[24+(NYN) * 24] = 1;
    W_e[25+(NYN) * 25] = 0.001;
    W_e[26+(NYN) * 26] = 0.001;
    W_e[27+(NYN) * 27] = 0.001;
    W_e[28+(NYN) * 28] = 0.001;
    W_e[29+(NYN) * 29] = 0.001;
    W_e[30+(NYN) * 30] = 0.001;
    W_e[31+(NYN) * 31] = 0.001;
    W_e[32+(NYN) * 32] = 0.001;
    W_e[33+(NYN) * 33] = 0.001;
    W_e[34+(NYN) * 34] = 0.001;
    W_e[35+(NYN) * 35] = 1;
    W_e[36+(NYN) * 36] = 0.001;
    W_e[37+(NYN) * 37] = 0.001;
    W_e[38+(NYN) * 38] = 0.001;
    W_e[39+(NYN) * 39] = 0.001;
    W_e[40+(NYN) * 40] = 0.001;
    W_e[41+(NYN) * 41] = 0.001;
    W_e[42+(NYN) * 42] = 0.001;
    W_e[43+(NYN) * 43] = 0.001;
    W_e[44+(NYN) * 44] = 0.001;
    W_e[45+(NYN) * 45] = 0.001;
    W_e[46+(NYN) * 46] = 10000;
    W_e[47+(NYN) * 47] = 10000;
    W_e[48+(NYN) * 48] = 10000;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e);
    free(W_e);
    double* Vx_e = calloc(NYN*NX, sizeof(double));
    // change only the non-zero elements:
    
    Vx_e[0+(NYN) * 0] = 1;
    Vx_e[1+(NYN) * 1] = 1;
    Vx_e[2+(NYN) * 2] = 1;
    Vx_e[3+(NYN) * 3] = 1;
    Vx_e[4+(NYN) * 4] = 1;
    Vx_e[5+(NYN) * 5] = 1;
    Vx_e[6+(NYN) * 6] = 1;
    Vx_e[7+(NYN) * 7] = 1;
    Vx_e[8+(NYN) * 8] = 1;
    Vx_e[9+(NYN) * 9] = 1;
    Vx_e[10+(NYN) * 10] = 1;
    Vx_e[11+(NYN) * 11] = 1;
    Vx_e[12+(NYN) * 12] = 1;
    Vx_e[13+(NYN) * 13] = 1;
    Vx_e[14+(NYN) * 14] = 1;
    Vx_e[15+(NYN) * 15] = 1;
    Vx_e[16+(NYN) * 16] = 1;
    Vx_e[17+(NYN) * 17] = 1;
    Vx_e[18+(NYN) * 18] = 1;
    Vx_e[19+(NYN) * 19] = 1;
    Vx_e[20+(NYN) * 20] = 1;
    Vx_e[21+(NYN) * 21] = 1;
    Vx_e[22+(NYN) * 22] = 1;
    Vx_e[23+(NYN) * 23] = 1;
    Vx_e[24+(NYN) * 24] = 1;
    Vx_e[25+(NYN) * 25] = 1;
    Vx_e[26+(NYN) * 26] = 1;
    Vx_e[27+(NYN) * 27] = 1;
    Vx_e[28+(NYN) * 28] = 1;
    Vx_e[29+(NYN) * 29] = 1;
    Vx_e[30+(NYN) * 30] = 1;
    Vx_e[31+(NYN) * 31] = 1;
    Vx_e[32+(NYN) * 32] = 1;
    Vx_e[33+(NYN) * 33] = 1;
    Vx_e[34+(NYN) * 34] = 1;
    Vx_e[35+(NYN) * 35] = 1;
    Vx_e[36+(NYN) * 36] = 1;
    Vx_e[37+(NYN) * 37] = 1;
    Vx_e[38+(NYN) * 38] = 1;
    Vx_e[39+(NYN) * 39] = 1;
    Vx_e[40+(NYN) * 40] = 1;
    Vx_e[41+(NYN) * 41] = 1;
    Vx_e[42+(NYN) * 42] = 1;
    Vx_e[43+(NYN) * 43] = 1;
    Vx_e[44+(NYN) * 44] = 1;
    Vx_e[45+(NYN) * 45] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Vx", Vx_e);
    free(Vx_e);



    /**** Constraints ****/

    // bounds for initial stage
    // x0
    int* idxbx0 = malloc(NBX0 * sizeof(int));
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;
    idxbx0[4] = 4;
    idxbx0[5] = 5;
    idxbx0[6] = 6;
    idxbx0[7] = 7;
    idxbx0[8] = 8;
    idxbx0[9] = 9;
    idxbx0[10] = 10;
    idxbx0[11] = 11;
    idxbx0[12] = 12;
    idxbx0[13] = 13;
    idxbx0[14] = 14;
    idxbx0[15] = 15;
    idxbx0[16] = 16;
    idxbx0[17] = 17;
    idxbx0[18] = 18;
    idxbx0[19] = 19;
    idxbx0[20] = 20;
    idxbx0[21] = 21;
    idxbx0[22] = 22;
    idxbx0[23] = 23;
    idxbx0[24] = 24;
    idxbx0[25] = 25;
    idxbx0[26] = 26;
    idxbx0[27] = 27;
    idxbx0[28] = 28;
    idxbx0[29] = 29;
    idxbx0[30] = 30;
    idxbx0[31] = 31;
    idxbx0[32] = 32;
    idxbx0[33] = 33;
    idxbx0[34] = 34;
    idxbx0[35] = 35;
    idxbx0[36] = 36;
    idxbx0[37] = 37;
    idxbx0[38] = 38;
    idxbx0[39] = 39;
    idxbx0[40] = 40;
    idxbx0[41] = 41;
    idxbx0[42] = 42;
    idxbx0[43] = 43;
    idxbx0[44] = 44;
    idxbx0[45] = 45;

    double* lubx0 = calloc(2*NBX0, sizeof(double));
    double* lbx0 = lubx0;
    double* ubx0 = lubx0 + NBX0;
    // change only the non-zero elements:
    lbx0[0] = 1.5707963267948966;
    ubx0[0] = 1.5707963267948966;
    lbx0[1] = 0.3141592653589793;
    ubx0[1] = 0.3141592653589793;
    lbx0[2] = -0.028201378200947976;
    ubx0[2] = -0.028201378200947976;
    lbx0[3] = -0.025423843544886696;
    ubx0[3] = -0.025423843544886696;
    lbx0[4] = -0.022773678035968668;
    ubx0[4] = -0.022773678035968668;
    lbx0[5] = -0.020256198193234612;
    ubx0[5] = -0.020256198193234612;
    lbx0[6] = -0.01787566099064464;
    ubx0[6] = -0.01787566099064464;
    lbx0[7] = -0.01563544779039782;
    ubx0[7] = -0.01563544779039782;
    lbx0[8] = -0.013538223698398973;
    ubx0[8] = -0.013538223698398973;
    lbx0[9] = -0.011586073537338032;
    ubx0[9] = -0.011586073537338032;
    lbx0[10] = -0.009780616183575582;
    ubx0[10] = -0.009780616183575582;
    lbx0[11] = -0.008123099353472996;
    ubx0[11] = -0.008123099353472996;
    lbx0[12] = -0.39269908169872414;
    ubx0[12] = -0.39269908169872414;
    lbx0[13] = -0.0066305212370094116;
    ubx0[13] = -0.0066305212370094116;
    lbx0[14] = -0.00530467521219187;
    ubx0[14] = -0.00530467521219187;
    lbx0[15] = -0.004127814134718859;
    ubx0[15] = -0.004127814134718859;
    lbx0[16] = -0.0030992169376013973;
    ubx0[16] = -0.0030992169376013973;
    lbx0[17] = -0.002218316178472931;
    ubx0[17] = -0.002218316178472931;
    lbx0[18] = -0.0014846862713757713;
    ubx0[18] = -0.0014846862713757713;
    lbx0[19] = -0.000898027075052084;
    ubx0[19] = -0.000898027075052084;
    lbx0[20] = -0.00045814416878752634;
    ubx0[20] = -0.00045814416878752634;
    lbx0[21] = -0.0001649269084287587;
    ubx0[21] = -0.0001649269084287587;
    lbx0[22] = -0.00001832512746249281;
    ubx0[22] = -0.00001832512746249281;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);
    free(idxbx0);
    free(lubx0);
    // idxbxe_0
    int* idxbxe_0 = malloc(46 * sizeof(int));
    
    idxbxe_0[0] = 0;
    idxbxe_0[1] = 1;
    idxbxe_0[2] = 2;
    idxbxe_0[3] = 3;
    idxbxe_0[4] = 4;
    idxbxe_0[5] = 5;
    idxbxe_0[6] = 6;
    idxbxe_0[7] = 7;
    idxbxe_0[8] = 8;
    idxbxe_0[9] = 9;
    idxbxe_0[10] = 10;
    idxbxe_0[11] = 11;
    idxbxe_0[12] = 12;
    idxbxe_0[13] = 13;
    idxbxe_0[14] = 14;
    idxbxe_0[15] = 15;
    idxbxe_0[16] = 16;
    idxbxe_0[17] = 17;
    idxbxe_0[18] = 18;
    idxbxe_0[19] = 19;
    idxbxe_0[20] = 20;
    idxbxe_0[21] = 21;
    idxbxe_0[22] = 22;
    idxbxe_0[23] = 23;
    idxbxe_0[24] = 24;
    idxbxe_0[25] = 25;
    idxbxe_0[26] = 26;
    idxbxe_0[27] = 27;
    idxbxe_0[28] = 28;
    idxbxe_0[29] = 29;
    idxbxe_0[30] = 30;
    idxbxe_0[31] = 31;
    idxbxe_0[32] = 32;
    idxbxe_0[33] = 33;
    idxbxe_0[34] = 34;
    idxbxe_0[35] = 35;
    idxbxe_0[36] = 36;
    idxbxe_0[37] = 37;
    idxbxe_0[38] = 38;
    idxbxe_0[39] = 39;
    idxbxe_0[40] = 40;
    idxbxe_0[41] = 41;
    idxbxe_0[42] = 42;
    idxbxe_0[43] = 43;
    idxbxe_0[44] = 44;
    idxbxe_0[45] = 45;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbxe", idxbxe_0);
    free(idxbxe_0);

    /* constraints that are the same for initial and intermediate */

    // ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxsbx", idxsbx);
    // ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lsbx", lsbx);
    // ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "usbx", usbx);

    // soft bounds on x
    int* idxsbx = malloc(NSBX * sizeof(int));
    idxsbx[0] = 0;
    idxsbx[1] = 1;
    idxsbx[2] = 2;

    double* lusbx = calloc(2*NSBX, sizeof(double));
    double* lsbx = lusbx;
    double* usbx = lusbx + NSBX;

    for (int i = 1; i < N; i++)
    {       
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxsbx", idxsbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lsbx", lsbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "usbx", usbx);
    }
    free(idxsbx);
    free(lusbx);
    // u
    int* idxbu = malloc(NBU * sizeof(int));
    
    idxbu[0] = 0;
    idxbu[1] = 1;
    idxbu[2] = 2;
    double* lubu = calloc(2*NBU, sizeof(double));
    double* lbu = lubu;
    double* ubu = lubu + NBU;
    
    lbu[0] = -20;
    ubu[0] = 20;
    lbu[1] = -10;
    ubu[1] = 10;
    lbu[2] = -10;
    ubu[2] = 10;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubu", ubu);
    }
    free(idxbu);
    free(lubu);








    // x
    int* idxbx = malloc(NBX * sizeof(int));
    
    idxbx[0] = 23;
    idxbx[1] = 24;
    idxbx[2] = 35;
    double* lubx = calloc(2*NBX, sizeof(double));
    double* lbx = lubx;
    double* ubx = lubx + NBX;
    
    lbx[0] = -2.5;
    ubx[0] = 2.5;
    lbx[1] = -2.5;
    ubx[1] = 2.5;
    lbx[2] = -2.5;
    ubx[2] = 2.5;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbx", idxbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbx", lbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubx", ubx);
    }
    free(idxbx);
    free(lubx);







    /* terminal constraints */

    // set up bounds for last stage
    // x
    int* idxbx_e = malloc(NBXN * sizeof(int));
    
    idxbx_e[0] = 23;
    idxbx_e[1] = 24;
    idxbx_e[2] = 35;
    double* lubx_e = calloc(2*NBXN, sizeof(double));
    double* lbx_e = lubx_e;
    double* ubx_e = lubx_e + NBXN;
    
    lbx_e[0] = -2.5;
    ubx_e[0] = 2.5;
    lbx_e[1] = -2.5;
    ubx_e[1] = 2.5;
    lbx_e[2] = -2.5;
    ubx_e[2] = 2.5;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "idxbx", idxbx_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lbx", lbx_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "ubx", ubx_e);
    free(idxbx_e);
    free(lubx_e);














}


/**
 * Internal function for flexible_arm_nq23_acados_create: step 6
 */
void flexible_arm_nq23_acados_create_6_set_opts(flexible_arm_nq23_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    void *nlp_opts = capsule->nlp_opts;

    /************************************************
    *  opts
    ************************************************/


    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization", "fixed_step");int full_step_dual = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "full_step_dual", &full_step_dual);
    // TODO: these options are lower level -> should be encapsulated! maybe through hessian approx option.
    bool output_z_val = true;
    bool sens_algebraic_val = true;

    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_output_z", &output_z_val);
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_sens_algebraic", &sens_algebraic_val);

    // set collocation type (relevant for implicit integrators)
    sim_collocation_type collocation_type = GAUSS_LEGENDRE;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_collocation_type", &collocation_type);

    // set up sim_method_num_steps
    // all sim_method_num_steps are identical
    int sim_method_num_steps = 2;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_steps", &sim_method_num_steps);

    // set up sim_method_num_stages
    // all sim_method_num_stages are identical
    int sim_method_num_stages = 2;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_stages", &sim_method_num_stages);

    int newton_iter_val = 3;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_iter", &newton_iter_val);


    // set up sim_method_jac_reuse
    bool tmp_bool = (bool) 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_jac_reuse", &tmp_bool);

    double nlp_solver_step_length = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "step_length", &nlp_solver_step_length);

    double levenberg_marquardt = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;

    const int qp_solver_cond_N_ori = 100;
    qp_solver_cond_N = N < qp_solver_cond_N_ori ? N : qp_solver_cond_N_ori; // use the minimum value here
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    int nlp_solver_ext_qp_res = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "BALANCE");



    int qp_solver_iter_max = 50;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);

int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);
    int qp_solver_cond_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);

    int qp_solver_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);


    int ext_cost_num_hess = 0;
}


/**
 * Internal function for flexible_arm_nq23_acados_create: step 7
 */
void flexible_arm_nq23_acados_create_7_set_nlp_out(flexible_arm_nq23_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;

    // initialize with x0
    
    x0[0] = 1.5707963267948966;
    x0[1] = 0.3141592653589793;
    x0[2] = -0.028201378200947976;
    x0[3] = -0.025423843544886696;
    x0[4] = -0.022773678035968668;
    x0[5] = -0.020256198193234612;
    x0[6] = -0.01787566099064464;
    x0[7] = -0.01563544779039782;
    x0[8] = -0.013538223698398973;
    x0[9] = -0.011586073537338032;
    x0[10] = -0.009780616183575582;
    x0[11] = -0.008123099353472996;
    x0[12] = -0.39269908169872414;
    x0[13] = -0.0066305212370094116;
    x0[14] = -0.00530467521219187;
    x0[15] = -0.004127814134718859;
    x0[16] = -0.0030992169376013973;
    x0[17] = -0.002218316178472931;
    x0[18] = -0.0014846862713757713;
    x0[19] = -0.000898027075052084;
    x0[20] = -0.00045814416878752634;
    x0[21] = -0.0001649269084287587;
    x0[22] = -0.00001832512746249281;


    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x0);
    free(xu0);
}


/**
 * Internal function for flexible_arm_nq23_acados_create: step 8
 */
//void flexible_arm_nq23_acados_create_8_create_solver(flexible_arm_nq23_solver_capsule* capsule)
//{
//    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts);
//}

/**
 * Internal function for flexible_arm_nq23_acados_create: step 9
 */
int flexible_arm_nq23_acados_create_9_precompute(flexible_arm_nq23_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int flexible_arm_nq23_acados_create_with_discretization(flexible_arm_nq23_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != FLEXIBLE_ARM_NQ23_N && !new_time_steps) {
        fprintf(stderr, "flexible_arm_nq23_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, FLEXIBLE_ARM_NQ23_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    flexible_arm_nq23_acados_create_1_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 3) create and set dimensions
    capsule->nlp_dims = flexible_arm_nq23_acados_create_2_create_and_set_dimensions(capsule);
    flexible_arm_nq23_acados_create_3_create_and_set_functions(capsule);

    // 4) set default parameters in functions
    flexible_arm_nq23_acados_create_4_set_default_parameters(capsule);

    // 5) create and set nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);
    flexible_arm_nq23_acados_create_5_set_nlp_in(capsule, N, new_time_steps);

    // 6) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    flexible_arm_nq23_acados_create_6_set_opts(capsule);

    // 7) create and set nlp_out
    // 7.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 7.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    flexible_arm_nq23_acados_create_7_set_nlp_out(capsule);

    // 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts);
    //flexible_arm_nq23_acados_create_8_create_solver(capsule);

    // 9) do precomputations
    int status = flexible_arm_nq23_acados_create_9_precompute(capsule);
    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int flexible_arm_nq23_acados_update_qp_solver_cond_N(flexible_arm_nq23_solver_capsule* capsule, int qp_solver_cond_N)
{
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from flexible_arm_nq23_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts);

    // -> 9) do precomputations
    int status = flexible_arm_nq23_acados_create_9_precompute(capsule);
    return status;
}


int flexible_arm_nq23_acados_reset(flexible_arm_nq23_solver_capsule* capsule, int reset_qp_solver_mem)
{

    // set initialization to all zeros

    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    int nx, nu, nv, ns, nz, ni, dim;

    double* buffer = calloc(NX+NU+NZ+2*NS+2*NSN+NBX+NBU+NG+NH+NPHI+NBX0+NBXN+NHN+NPHIN+NGN, sizeof(double));

    for(int i=0; i<N+1; i++)
    {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "sl", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "su", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "lam", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "t", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "z", buffer);
        if (i<N)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "pi", buffer);
            ocp_nlp_set(nlp_config, nlp_solver, i, "xdot_guess", buffer);
            ocp_nlp_set(nlp_config, nlp_solver, i, "z_guess", buffer);
        
        }
    }
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }

    free(buffer);
    return 0;
}




int flexible_arm_nq23_acados_update_params(flexible_arm_nq23_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 0;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }

    return solver_status;
}


int flexible_arm_nq23_acados_update_params_sparse(flexible_arm_nq23_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    int solver_status = 0;

    int casadi_np = 0;
    if (casadi_np < n_update) {
        printf("flexible_arm_nq23_acados_update_params_sparse: trying to set %d parameters for external functions."
            " External function has %d parameters. Exiting.\n", n_update, casadi_np);
        exit(1);
    }
    // for (int i = 0; i < n_update; i++)
    // {
    //     if (idx[i] > casadi_np) {
    //         printf("flexible_arm_nq23_acados_update_params_sparse: attempt to set parameters with index %d, while"
    //             " external functions only has %d parameters. Exiting.\n", idx[i], casadi_np);
    //         exit(1);
    //     }
    //     printf("param %d value %e\n", idx[i], p[i]);
    // }

    return 0;
}

int flexible_arm_nq23_acados_solve(flexible_arm_nq23_solver_capsule* capsule)
{
    // solve NLP 
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}


int flexible_arm_nq23_acados_free(flexible_arm_nq23_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < N; i++)
    {
        external_function_param_casadi_free(&capsule->impl_dae_fun[i]);
        external_function_param_casadi_free(&capsule->impl_dae_fun_jac_x_xdot_z[i]);
        external_function_param_casadi_free(&capsule->impl_dae_jac_x_xdot_u_z[i]);
    }
    free(capsule->impl_dae_fun);
    free(capsule->impl_dae_fun_jac_x_xdot_z);
    free(capsule->impl_dae_jac_x_xdot_u_z);

    // cost

    // constraints

    return 0;
}

ocp_nlp_in *flexible_arm_nq23_acados_get_nlp_in(flexible_arm_nq23_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *flexible_arm_nq23_acados_get_nlp_out(flexible_arm_nq23_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *flexible_arm_nq23_acados_get_sens_out(flexible_arm_nq23_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *flexible_arm_nq23_acados_get_nlp_solver(flexible_arm_nq23_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *flexible_arm_nq23_acados_get_nlp_config(flexible_arm_nq23_solver_capsule* capsule) { return capsule->nlp_config; }
void *flexible_arm_nq23_acados_get_nlp_opts(flexible_arm_nq23_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *flexible_arm_nq23_acados_get_nlp_dims(flexible_arm_nq23_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *flexible_arm_nq23_acados_get_nlp_plan(flexible_arm_nq23_solver_capsule* capsule) { return capsule->nlp_solver_plan; }


void flexible_arm_nq23_acados_print_stats(flexible_arm_nq23_solver_capsule* capsule)
{
    int sqp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "sqp_iter", &sqp_iter);
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "stat_m", &stat_m);

    
    double stat[600];
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "statistics", stat);

    int nrow = sqp_iter+1 < stat_m ? sqp_iter+1 : stat_m;

    printf("iter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tqp_stat\tqp_iter\talpha");
    if (stat_n > 8)
        printf("\t\tqp_res_stat\tqp_res_eq\tqp_res_ineq\tqp_res_comp");
    printf("\n");
    printf("iter\tqp_stat\tqp_iter\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            tmp_int = (int) stat[i + j * nrow];
            printf("%d\t", tmp_int);
        }
        printf("\n");
    }
}

