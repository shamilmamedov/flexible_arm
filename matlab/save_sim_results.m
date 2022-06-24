clc; close all;% clear all;


t = q_a.Time;
q1 = q_a.Data;
q2 = q_p1.Data;
q3 = q_p2.Data;

dq1 = dq_a.Data;
dq2 = dq_p1.Data;
dq3 = dq_p2.Data;

tau = tau_a.Data;

T = table(t, q1, q2, q3, dq1, dq2, dq3, tau);

writetable(T, 'data/matlab_sim_3s.csv')

% figure
% plot(t, dq2)
% hold on
% plot(t, dq3)
% legend('qp1', 'qp2')
% grid on

% figure
% plot(t, tau)
% grid on