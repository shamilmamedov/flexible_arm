#include <iostream>
#include <cmath>

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
 
#include "pinocchio/autodiff/casadi.hpp"
#include <casadi/casadi.hpp>

int main()
{
    // Short name for convenience
    namespace pin = pinocchio;
    
    typedef double Scalar;
    typedef casadi::SX ADScalar;

    typedef pinocchio::ModelTpl<Scalar> Model;
    typedef Model::Data Data;

    typedef pin::ModelTpl<ADScalar> ADModel;
    typedef ADModel::Data ADData;

    typedef Model::ConfigVectorType ConfigVector;
    typedef Model::TangentVectorType TangentVector;

    typedef ADModel::ConfigVectorType ConfigVectorAD;
    typedef ADModel::TangentVectorType TangentVectorAD;

    // Path to URDF file
    const std::string urdf_filename = "/home/shamil/Desktop/phd/code/"
                                    "flexible_arm/models/flexible_arm_v1.urdf";

    // Load the urdf model
    Model model;
    pin::urdf::buildModel(urdf_filename, model);
    std::cout << "model name: " << model.name << std::endl;

    // Create data required by the algorithms
    Data data(model);

    // Get EE frame ID for forward kinematics
    const std::string ee_link_name = "link5_to_load";
    Model::Index ee_frame_id = model.getFrameId(ee_link_name);
    std::cout << "EE frame ID: " << ee_frame_id << std::endl;

    // Set a configuration, velocity and torque
    ConfigVector q(model.nq);
    q = pin::randomConfiguration(model);
    std::cout << "q: " << q.transpose() << std::endl;
    TangentVector v(TangentVector::Random(model.nv));
    TangentVector tau(TangentVector::Random(model.nv));

    pin::aba(model,data,q,v,tau);

    // Define symbolic model and symbolic variables
    ADModel ad_model = model.cast<ADScalar>();
    ADData ad_data(ad_model);

    casadi::SX cs_q = casadi::SX::sym("q", model.nq);
    ConfigVectorAD q_ad(model.nq);
    q_ad = Eigen::Map<ConfigVectorAD>(static_cast< std::vector<ADScalar> >(cs_q).data(),model.nq,1);

    casadi::SX cs_v = casadi::SX::sym("v", model.nv);
    TangentVectorAD v_ad(model.nv);
    v_ad = Eigen::Map<TangentVectorAD>(static_cast< std::vector<ADScalar> >(cs_v).data(),model.nv,1);
    
    casadi::SX cs_tau = casadi::SX::sym("tau", model.nv);
    TangentVectorAD tau_ad(model.nv);
    tau_ad = Eigen::Map<TangentVectorAD>(static_cast< std::vector<ADScalar> >(cs_tau).data(),model.nv,1);

    // Compute forward dynamics for symbolic model
    aba(ad_model,ad_data,q_ad,v_ad,tau_ad);
    // Create a casadi function for forward dynamics
    casadi::SX cs_ddq(model.nv,1);
    for(Eigen::DenseIndex k = 0; k < model.nv; ++k)
        cs_ddq(k) = ad_data.ddq[k];
    casadi::Function eval_aba("eval_aba",
                              casadi::SXVector {cs_q, cs_v, cs_tau},
                              casadi::SXVector {cs_ddq});
    eval_aba.save("aba.casadi");

    // Evaluate forward dynamics and compare with numerical solution
    std::vector<double> q_vec((size_t)model.nq);
    Eigen::Map<ConfigVector>(q_vec.data(),model.nq,1) = q;
    
    std::vector<double> v_vec((size_t)model.nv);
    Eigen::Map<TangentVector>(v_vec.data(),model.nv,1) = v;
    
    std::vector<double> tau_vec((size_t)model.nv);
    Eigen::Map<TangentVector>(tau_vec.data(),model.nv,1) = tau;

    casadi::DM ddq_res = eval_aba(casadi::DMVector {q_vec, v_vec, tau_vec})[0];
    Data::TangentVectorType ddq_mat = Eigen::Map<Data::TangentVectorType>(static_cast< std::vector<double> >(ddq_res).data(),
                                                            model.nv,1);

    std::cout << "ddq numerical = " << data.ddq << std::endl;
    std::cout << "ddq casadi = " << ddq_mat << std::endl;

    // // Perform the forward kinematics over the kinematic tree
    // pin::forwardKinematics(model, data, q);
    // pin::updateFramePlacement(model, data, ee_frame_id);
 
    // Eigen::VectorXd p_ee = data.oMf[ee_frame_id].translation();
    // std::cout << "EE position: " << p_ee.transpose() << std::endl;

    // // Compute Jacobians
    // pin::computeJointJacobians(model, data, q);
    // pin::updateFramePlacement(model, data, ee_frame_id);
    // pin::Data::Matrix6x Jee_0(6, model.nv);
    // pin::getFrameJacobian(model, data, ee_frame_id, pin::LOCAL_WORLD_ALIGNED, Jee_0);
    // std::cout << Jee_0 << std::endl;

    // // FK with casadi
    // ADModel ad_model = model.cast<ADScalar>();
    // ADData ad_data(ad_model);

    // casadi::SX cs_q = casadi::SX::sym("q", model.nq);
    // ConfigVectorAD q_ad(model.nq);
    // pin::casadi::copy(cs_q, q_ad);

    // pin::forwardKinematics(ad_model, ad_data, q_ad);
    // pinocchio::updateGlobalPlacements(ad_model, ad_data);
    // pinocchio::updateFramePlacements(ad_model, ad_data);

    // casadi::SX cs_p_ee(3,1);
    // for(Eigen::DenseIndex k = 0; k < 3; ++k)
    //     cs_p_ee(k) = ad_data.oMf[ee_frame_id].translation()[k];

    // casadi::Function eval_fkp("eval_fkp",
    //                         casadi::SXVector {cs_q},
    //                         casadi::SXVector {cs_p_ee});

    // ConfigVector qq(model.nq);
    // qq << 0.0, 0.0, 0.0, 0.0, 0.0;

    // std::vector<double> q_vec((size_t)model.nq);
    // Eigen::Map<ConfigVector>(q_vec.data(),model.nq,1) = qq;

    // std::cout << eval_fkp(casadi::DMVector {q_vec}) << std::endl;

  return 0;
}
