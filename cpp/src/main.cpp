#include <iostream>
#include <cmath>
#include <map>

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
 
#include "pinocchio/autodiff/casadi.hpp"
#include <casadi/casadi.hpp>

int main(int argc,char* argv[])
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
    std::map<int,std::string> my_map = {
        {0, "zero"},
        {2, "two"},
        {1, "one"},
        {3, "three"},
        {5, "five"},
        {10, "ten"}
    };

    int n_dof = 3;
    int n_seg = 0;
    if (argc == 2){
        n_seg = std::stoi(argv[1]);
    } else {
        std::cout << "Select number of segments:" << std::endl;
        std::cin >> n_seg;
    }

    const std::string model_folder = "/home/shamil/Desktop/phd/code/"
                                    "flexible_arm/models/" + my_map.at(n_dof) + "_dof/" 
                                    + my_map.at(n_seg) + "_segments/";

    const std::string urdf_path = model_folder + "flexible_arm_" + std::to_string(n_dof) 
                                    + "dof_" + std::to_string(n_seg) + "s.urdf";

    // Load the urdf model
    Model model;
    pin::urdf::buildModel(urdf_path, model);
    std::cout << "model name: " << model.name << std::endl;

    // Create data required by the algorithms
    Data data(model);

    // Get EE frame ID for forward kinematics
    // const std::string ee_link_name = "virtual_link" + std::to_string(n_seg+1) + "_to_load";
    const std::string ee_link_name = "load";
    Model::Index ee_frame_id = model.getFrameId(ee_link_name);
    std::cout << "EE frame ID: " << ee_frame_id << std::endl;

    // Set a configuration, velocity and torque
    ConfigVector q(model.nq);
    q = pin::randomConfiguration(model);
    std::cout << "q: " << q.transpose() << std::endl;
    TangentVector v(TangentVector::Random(model.nv));
    TangentVector a(TangentVector::Random(model.nv));
    TangentVector tau(TangentVector::Random(model.nv));

    // Run ABA for a numerical model
    pin::aba(model, data, q, v, tau);

    // Define symbolic model and symbolic variables
    ADModel ad_model = model.cast<ADScalar>();
    ADData ad_data(ad_model);

    casadi::SX cs_q = casadi::SX::sym("q", model.nq);
    ConfigVectorAD q_ad(model.nq);
    q_ad = Eigen::Map<ConfigVectorAD>(static_cast< std::vector<ADScalar> >(cs_q).data(),model.nq,1);

    casadi::SX cs_v = casadi::SX::sym("v", model.nv);
    TangentVectorAD v_ad(model.nv);
    v_ad = Eigen::Map<TangentVectorAD>(static_cast< std::vector<ADScalar> >(cs_v).data(),model.nv,1);

    casadi::SX cs_a = casadi::SX::sym("a", model.nv);
    TangentVectorAD a_ad(model.nv);
    a_ad = Eigen::Map<TangentVectorAD>(static_cast< std::vector<ADScalar> >(cs_a).data(),model.nv,1);
    
    casadi::SX cs_tau = casadi::SX::sym("tau", model.nv);
    TangentVectorAD tau_ad(model.nv);
    tau_ad = Eigen::Map<TangentVectorAD>(static_cast< std::vector<ADScalar> >(cs_tau).data(),model.nv,1);

    // Compute forward dynamics for symbolic model
    aba(ad_model, ad_data, q_ad, v_ad, tau_ad);
    // Create a casadi function for forward dynamics
    casadi::SX cs_ddq(model.nv,1);
    for(Eigen::DenseIndex k = 0; k < model.nv; ++k)
        cs_ddq(k) = ad_data.ddq[k];
    casadi::Function eval_aba("eval_aba",
                              casadi::SXVector {cs_q, cs_v, cs_tau},
                              casadi::SXVector {cs_ddq});
    eval_aba.save(model_folder + "aba.casadi");

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


    // Perform the forward kinematics over the kinematic tree
    pin::forwardKinematics(model, data, q);
    pin::updateFramePlacement(model, data, ee_frame_id);
 
    Eigen::VectorXd p_ee = data.oMf[ee_frame_id].translation();
    auto v_ee =  pin::getFrameVelocity(model, data, 
                            ee_frame_id, pin::LOCAL_WORLD_ALIGNED);

    std::cout << "EE position: " << p_ee.transpose() << std::endl;
    std::cout << "EE velocity num: " << v_ee.linear().transpose()  << std::endl;

    // Compute Jacobians
    pin::computeJointJacobians(model, data, q);
    pin::updateFramePlacement(model, data, ee_frame_id);
    pin::Data::Matrix6x Jee_0(6, model.nv);
    pin::getFrameJacobian(model, data, ee_frame_id, pin::LOCAL_WORLD_ALIGNED, Jee_0);
    std::cout << "EE Jacobian num: " << std::endl;
    std::cout << Jee_0 << std::endl;

    // Forward kinematics with symbolic model
    pin::forwardKinematics(ad_model, ad_data, q_ad, v_ad, a_ad);
    pinocchio::updateGlobalPlacements(ad_model, ad_data);
    pinocchio::updateFramePlacements(ad_model, ad_data);

    auto ad_v_ee =  pin::getFrameVelocity(ad_model, ad_data, 
                          ee_frame_id, pin::LOCAL_WORLD_ALIGNED);

    // EE Jacobian
    pin::computeJointJacobians(ad_model, ad_data, q_ad);
    pin::computeJointJacobiansTimeVariation(ad_model, ad_data, q_ad, v_ad);
    pin::updateGlobalPlacements(ad_model, ad_data);
    pin::updateFramePlacements(ad_model, ad_data);

    auto ad_jac_ee = pin::getFrameJacobian(ad_model, ad_data, 
                        ee_frame_id, pin::LOCAL_WORLD_ALIGNED);

    ADData::Matrix6x ad_djac_ee(6, ad_model.nv); ad_djac_ee.fill(0.);
    pin::getFrameJacobianTimeVariation(ad_model, ad_data, 
                          ee_frame_id, pin::LOCAL_WORLD_ALIGNED, ad_djac_ee);

    // Get a symbolic expression for kinematic variables
    casadi::SX cs_p_ee(3,1);
    casadi::SX cs_v_ee(3,1);
    for(Eigen::DenseIndex k = 0; k < 3; ++k){
        cs_p_ee(k) = ad_data.oMf[ee_frame_id].translation()[k];
        cs_v_ee(k) = ad_v_ee.linear()[k];
    }

    casadi::SX cs_jac_ee(3, model.nv);
    casadi::SX cs_djac_ee(3, model.nv);
    for (Eigen::DenseIndex k = 0; k < 3; ++k){
        for (Eigen::DenseIndex j = 0; j < model.nv; ++j){
            cs_jac_ee(k,j) = ad_jac_ee(k,j);
            cs_djac_ee(k,j) = ad_djac_ee(k,j);
        }
    }

    // Create casadi functions for evaluating kinematics
    casadi::Function eval_fkp("eval_fkp",
                            casadi::SXVector {cs_q},
                            casadi::SXVector {cs_p_ee});
    eval_fkp.save(model_folder + "fkp.casadi");

    casadi::Function eval_vee("eval_vee",
                            casadi::SXVector {cs_q, cs_v},
                            casadi::SXVector {cs_v_ee});
    eval_vee.save(model_folder + "fkv.casadi");

    casadi::Function eval_Jee("eval_Jee",
                            casadi::SXVector {cs_q},
                            casadi::SXVector {cs_jac_ee});
    eval_Jee.save(model_folder + "fkJ.casadi");

    std::cout << "EE position sym: " << eval_fkp(casadi::DMVector {q_vec}) << std::endl;
    std::cout << "EE velocity sym: " << eval_vee(casadi::DMVector {q_vec, v_vec}) << std::endl;
    std::cout << "EE Jacobian sym: " << eval_Jee(casadi::DMVector {q_vec}) << std::endl;

    // ConfigVector qq(model.nq);
    // qq << 0.0, 0.0, 0.0, 0.0, 0.0;

    // std::vector<double> q_vec((size_t)model.nq);
    // Eigen::Map<ConfigVector>(q_vec.data(),model.nq,1) = qq;

    // std::cout << eval_fkp(casadi::DMVector {q_vec}) << std::endl;

  return 0;
}
