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

#ifndef ACADOS_SOLVER_flexible_arm_nq5_H_
#define ACADOS_SOLVER_flexible_arm_nq5_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define FLEXIBLE_ARM_NQ5_NX     10
#define FLEXIBLE_ARM_NQ5_NZ     3
#define FLEXIBLE_ARM_NQ5_NU     3
#define FLEXIBLE_ARM_NQ5_NP     0
#define FLEXIBLE_ARM_NQ5_NBX    3
#define FLEXIBLE_ARM_NQ5_NBX0   10
#define FLEXIBLE_ARM_NQ5_NBU    3
#define FLEXIBLE_ARM_NQ5_NSBX   3
#define FLEXIBLE_ARM_NQ5_NSBU   0
#define FLEXIBLE_ARM_NQ5_NSH    0
#define FLEXIBLE_ARM_NQ5_NSG    0
#define FLEXIBLE_ARM_NQ5_NSPHI  0
#define FLEXIBLE_ARM_NQ5_NSHN   0
#define FLEXIBLE_ARM_NQ5_NSGN   0
#define FLEXIBLE_ARM_NQ5_NSPHIN 0
#define FLEXIBLE_ARM_NQ5_NSBXN  0
#define FLEXIBLE_ARM_NQ5_NS     3
#define FLEXIBLE_ARM_NQ5_NSN    0
#define FLEXIBLE_ARM_NQ5_NG     0
#define FLEXIBLE_ARM_NQ5_NBXN   3
#define FLEXIBLE_ARM_NQ5_NGN    0
#define FLEXIBLE_ARM_NQ5_NY0    16
#define FLEXIBLE_ARM_NQ5_NY     16
#define FLEXIBLE_ARM_NQ5_NYN    13
#define FLEXIBLE_ARM_NQ5_N      100
#define FLEXIBLE_ARM_NQ5_NH     0
#define FLEXIBLE_ARM_NQ5_NPHI   0
#define FLEXIBLE_ARM_NQ5_NHN    0
#define FLEXIBLE_ARM_NQ5_NPHIN  0
#define FLEXIBLE_ARM_NQ5_NR     0

#ifdef __cplusplus
extern "C" {
#endif

// ** capsule for solver data **
typedef struct flexible_arm_nq5_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */
    // dynamics

    external_function_param_casadi *impl_dae_fun;
    external_function_param_casadi *impl_dae_fun_jac_x_xdot_z;
    external_function_param_casadi *impl_dae_jac_x_xdot_u_z;




    // cost






    // constraints




} flexible_arm_nq5_solver_capsule;

ACADOS_SYMBOL_EXPORT flexible_arm_nq5_solver_capsule * flexible_arm_nq5_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int flexible_arm_nq5_acados_free_capsule(flexible_arm_nq5_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int flexible_arm_nq5_acados_create(flexible_arm_nq5_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int flexible_arm_nq5_acados_reset(flexible_arm_nq5_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of flexible_arm_nq5_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int flexible_arm_nq5_acados_create_with_discretization(flexible_arm_nq5_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int flexible_arm_nq5_acados_update_time_steps(flexible_arm_nq5_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int flexible_arm_nq5_acados_update_qp_solver_cond_N(flexible_arm_nq5_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int flexible_arm_nq5_acados_update_params(flexible_arm_nq5_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int flexible_arm_nq5_acados_update_params_sparse(flexible_arm_nq5_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);

ACADOS_SYMBOL_EXPORT int flexible_arm_nq5_acados_solve(flexible_arm_nq5_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int flexible_arm_nq5_acados_free(flexible_arm_nq5_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void flexible_arm_nq5_acados_print_stats(flexible_arm_nq5_solver_capsule * capsule);
                     
ACADOS_SYMBOL_EXPORT ocp_nlp_in *flexible_arm_nq5_acados_get_nlp_in(flexible_arm_nq5_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *flexible_arm_nq5_acados_get_nlp_out(flexible_arm_nq5_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *flexible_arm_nq5_acados_get_sens_out(flexible_arm_nq5_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *flexible_arm_nq5_acados_get_nlp_solver(flexible_arm_nq5_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *flexible_arm_nq5_acados_get_nlp_config(flexible_arm_nq5_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *flexible_arm_nq5_acados_get_nlp_opts(flexible_arm_nq5_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *flexible_arm_nq5_acados_get_nlp_dims(flexible_arm_nq5_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *flexible_arm_nq5_acados_get_nlp_plan(flexible_arm_nq5_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_flexible_arm_nq5_H_
