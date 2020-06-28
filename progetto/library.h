#include "ROOT/RVec.hxx"
#include "ROOT/RDataFrame.hxx"
#include "Math/Vector4D.h"

using rvec = const ROOT::VecOps::RVec<float> &;
const auto z = 91.2;

float dr_def( rvec part_eta, rvec part_phi)
{
    float part_dr = ROOT::VecOps::DeltaR(part_eta[0], part_eta[1], part_phi[0], part_phi[1]);
    return part_dr;
}

bool dr_cut( rvec mu_eta, rvec mu_phi, rvec el_eta, rvec el_phi )
{
    auto mu_dr = ROOT::VecOps::DeltaR(mu_eta[0], mu_eta[1], mu_phi[0], mu_phi[1]);
    auto el_dr = ROOT::VecOps::DeltaR(el_eta[0], el_eta[1], el_phi[0], el_phi[1]);
    if (mu_dr < 0.02 || el_dr < 0.02){
        return false;
    }
    return true;
}

bool pt_cut(rvec mu_pt, rvec el_pt)
{
    auto mu_pt_sorted = ROOT::VecOps::Reverse(ROOT::VecOps::Sort(mu_pt));
    if (mu_pt_sorted[0] > 20 && mu_pt_sorted[1] > 10){
        return true;
    }
    auto el_pt_sorted = ROOT::VecOps::Reverse(ROOT::VecOps::Sort(el_pt));
    if (el_pt_sorted[0] > 20 && el_pt_sorted[1] > 10){
        return true;
    }
    return false;
}

ROOT::VecOps::RVec<float> z_mass(rvec el_pt, rvec el_eta, rvec el_phi, rvec el_mass,
                                 rvec mu_pt, rvec mu_eta, rvec mu_phi, rvec mu_mass)
{
    ROOT::Math::PtEtaPhiMVector p1(mu_pt[0], mu_eta[0], mu_phi[0], mu_mass[0]);
    ROOT::Math::PtEtaPhiMVector p2(mu_pt[1], mu_eta[1], mu_phi[1], mu_mass[1]);
    ROOT::Math::PtEtaPhiMVector p3(el_pt[0], el_eta[0], el_phi[0], el_mass[0]);
    ROOT::Math::PtEtaPhiMVector p4(el_pt[1], el_eta[1], el_phi[1], el_mass[1]);
    auto mu_z = (p1 + p2).M();
    auto el_z = (p3 + p4).M();
    ROOT::VecOps::RVec<float> z_masses(2);
    if(std::abs(mu_z - z) < std::abs(el_z - z)){
        z_masses[0] = mu_z;
        z_masses[1] = el_z;
    } else{
        z_masses[0] = el_z;
        z_masses[1] = mu_z;
    }
    return z_masses;
}

// Compute Higgs mass from two electrons and two muons
float h_mass(rvec el_pt, rvec el_eta, rvec el_phi, rvec el_mass, rvec mu_pt, rvec mu_eta,
                                rvec mu_phi, rvec mu_mass)
{
    ROOT::Math::PtEtaPhiMVector p1(mu_pt[0], mu_eta[0], mu_phi[0], mu_mass[0]);
    ROOT::Math::PtEtaPhiMVector p2(mu_pt[1], mu_eta[1], mu_phi[1], mu_mass[1]);
    ROOT::Math::PtEtaPhiMVector p3(el_pt[0], el_eta[0], el_phi[0], el_mass[0]);
    ROOT::Math::PtEtaPhiMVector p4(el_pt[1], el_eta[1], el_phi[1], el_mass[1]);
    return (p1 + p2 + p3 + p4).M();
}
