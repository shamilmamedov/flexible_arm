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
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_flexible_arm_nq23.h"

#define NX     FLEXIBLE_ARM_NQ23_NX
#define NZ     FLEXIBLE_ARM_NQ23_NZ
#define NU     FLEXIBLE_ARM_NQ23_NU
#define NP     FLEXIBLE_ARM_NQ23_NP


int main()
{
    int status = 0;
    sim_solver_capsule *capsule = flexible_arm_nq23_acados_sim_solver_create_capsule();
    status = flexible_arm_nq23_acados_sim_create(capsule);

    if (status)
    {
        printf("acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    sim_config *acados_sim_config = flexible_arm_nq23_acados_get_sim_config(capsule);
    sim_in *acados_sim_in = flexible_arm_nq23_acados_get_sim_in(capsule);
    sim_out *acados_sim_out = flexible_arm_nq23_acados_get_sim_out(capsule);
    void *acados_sim_dims = flexible_arm_nq23_acados_get_sim_dims(capsule);

    // initial condition
    double x_current[NX];
    x_current[0] = 0.0;
    x_current[1] = 0.0;
    x_current[2] = 0.0;
    x_current[3] = 0.0;
    x_current[4] = 0.0;
    x_current[5] = 0.0;
    x_current[6] = 0.0;
    x_current[7] = 0.0;
    x_current[8] = 0.0;
    x_current[9] = 0.0;
    x_current[10] = 0.0;
    x_current[11] = 0.0;
    x_current[12] = 0.0;
    x_current[13] = 0.0;
    x_current[14] = 0.0;
    x_current[15] = 0.0;
    x_current[16] = 0.0;
    x_current[17] = 0.0;
    x_current[18] = 0.0;
    x_current[19] = 0.0;
    x_current[20] = 0.0;
    x_current[21] = 0.0;
    x_current[22] = 0.0;
    x_current[23] = 0.0;
    x_current[24] = 0.0;
    x_current[25] = 0.0;
    x_current[26] = 0.0;
    x_current[27] = 0.0;
    x_current[28] = 0.0;
    x_current[29] = 0.0;
    x_current[30] = 0.0;
    x_current[31] = 0.0;
    x_current[32] = 0.0;
    x_current[33] = 0.0;
    x_current[34] = 0.0;
    x_current[35] = 0.0;
    x_current[36] = 0.0;
    x_current[37] = 0.0;
    x_current[38] = 0.0;
    x_current[39] = 0.0;
    x_current[40] = 0.0;
    x_current[41] = 0.0;
    x_current[42] = 0.0;
    x_current[43] = 0.0;
    x_current[44] = 0.0;
    x_current[45] = 0.0;

  
    x_current[0] = 1.5707963267948966;
    x_current[1] = 0.3141592653589793;
    x_current[2] = -0.028201378200947976;
    x_current[3] = -0.025423843544886696;
    x_current[4] = -0.022773678035968668;
    x_current[5] = -0.020256198193234612;
    x_current[6] = -0.01787566099064464;
    x_current[7] = -0.01563544779039782;
    x_current[8] = -0.013538223698398973;
    x_current[9] = -0.011586073537338032;
    x_current[10] = -0.009780616183575582;
    x_current[11] = -0.008123099353472996;
    x_current[12] = -0.39269908169872414;
    x_current[13] = -0.0066305212370094116;
    x_current[14] = -0.00530467521219187;
    x_current[15] = -0.004127814134718859;
    x_current[16] = -0.0030992169376013973;
    x_current[17] = -0.002218316178472931;
    x_current[18] = -0.0014846862713757713;
    x_current[19] = -0.000898027075052084;
    x_current[20] = -0.00045814416878752634;
    x_current[21] = -0.0001649269084287587;
    x_current[22] = -0.00001832512746249281;
    x_current[23] = 0;
    x_current[24] = 0;
    x_current[25] = 0;
    x_current[26] = 0;
    x_current[27] = 0;
    x_current[28] = 0;
    x_current[29] = 0;
    x_current[30] = 0;
    x_current[31] = 0;
    x_current[32] = 0;
    x_current[33] = 0;
    x_current[34] = 0;
    x_current[35] = 0;
    x_current[36] = 0;
    x_current[37] = 0;
    x_current[38] = 0;
    x_current[39] = 0;
    x_current[40] = 0;
    x_current[41] = 0;
    x_current[42] = 0;
    x_current[43] = 0;
    x_current[44] = 0;
    x_current[45] = 0;
    
  


    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;
    u0[1] = 0.0;
    u0[2] = 0.0;

    int n_sim_steps = 3;
    // solve ocp in loop
    for (int ii = 0; ii < n_sim_steps; ii++)
    {
        sim_in_set(acados_sim_config, acados_sim_dims,
            acados_sim_in, "x", x_current);
        status = flexible_arm_nq23_acados_sim_solve(capsule);

        if (status != ACADOS_SUCCESS)
        {
            printf("acados_solve() failed with status %d.\n", status);
        }

        sim_out_get(acados_sim_config, acados_sim_dims,
               acados_sim_out, "x", x_current);
        
        printf("\nx_current, %d\n", ii);
        for (int jj = 0; jj < NX; jj++)
        {
            printf("%e\n", x_current[jj]);
        }
    }

    printf("\nPerformed %d simulation steps with acados integrator successfully.\n\n", n_sim_steps);

    // free solver
    status = flexible_arm_nq23_acados_sim_free(capsule);
    if (status) {
        printf("flexible_arm_nq23_acados_sim_free() returned status %d. \n", status);
    }

    flexible_arm_nq23_acados_sim_solver_free_capsule(capsule);

    return status;
}
