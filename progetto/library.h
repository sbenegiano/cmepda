#include "ROOT/RVec.hxx"
#include "ROOT/RDataFrame.hxx"
#include "TLorentzVector.h"
#include "TMath.h"
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
float h_mass(rvec el_pt, rvec el_eta, rvec el_phi, rvec el_mass, 
             rvec mu_pt, rvec mu_eta, rvec mu_phi, rvec mu_mass)
{
    ROOT::Math::PtEtaPhiMVector p1(mu_pt[0], mu_eta[0], mu_phi[0], mu_mass[0]);
    ROOT::Math::PtEtaPhiMVector p2(mu_pt[1], mu_eta[1], mu_phi[1], mu_mass[1]);
    ROOT::Math::PtEtaPhiMVector p3(el_pt[0], el_eta[0], el_phi[0], el_mass[0]);
    ROOT::Math::PtEtaPhiMVector p4(el_pt[1], el_eta[1], el_phi[1], el_mass[1]);
    return (p1 + p2 + p3 + p4).M();
}

float costheta_star(rvec el_pt, rvec el_eta, rvec el_phi, rvec el_mass,
                 rvec mu_pt, rvec mu_eta, rvec mu_phi, rvec mu_mass)
{
    TLorentzVector mu1, mu2, el1, el2, z1, z2, H;
    TVector3 beta_H;

    mu1.SetPtEtaPhiM(mu_pt[0], mu_eta[0], mu_phi[0], mu_mass[0]);
    mu2.SetPtEtaPhiM(mu_pt[1], mu_eta[1], mu_phi[1], mu_mass[1]);
    el1.SetPtEtaPhiM(el_pt[0], el_eta[0], el_phi[0], el_mass[0]);
    el2.SetPtEtaPhiM(el_pt[1], el_eta[1], el_phi[1], el_mass[1]);
    auto mu_z = (mu1 + mu2);
    auto el_z = (el1 + el2);
    if(std::abs(mu_z.M() - z) < std::abs(el_z.M() - z)){
        z1 = mu_z;
        z2 = el_z;
    } else{
        z1 = el_z;
        z2 = mu_z;
    }

    H = z1 + z2;
    beta_H = H.BoostVector();
    z1.Boost(-beta_H);
    float costheta_star = z1.CosTheta();
    return costheta_star;
}

ROOT::VecOps::RVec<float> costheta(rvec el_pt, rvec el_eta, rvec el_phi, rvec el_mass, rvec el_charge,
                                   rvec mu_pt, rvec mu_eta, rvec mu_phi, rvec mu_mass, rvec mu_charge)
{
    TLorentzVector el1, el2, mu1, mu2, z1, z2, lep1_n, lep2_n;
    TVector3 beta_z1, beta_z2, beta_lep1, beta_lep2;
    ROOT::VecOps::RVec<float> costheta_vector(2);

    mu1.SetPtEtaPhiM(mu_pt[0], mu_eta[0], mu_phi[0], mu_mass[0]);
    mu2.SetPtEtaPhiM(mu_pt[1], mu_eta[1], mu_phi[1], mu_mass[1]);
    el1.SetPtEtaPhiM(el_pt[0], el_eta[0], el_phi[0], el_mass[0]);
    el2.SetPtEtaPhiM(el_pt[1], el_eta[1], el_phi[1], el_mass[1]);

    auto mu_z = (mu1 + mu2);
    auto el_z = (el1 + el2);
    if(std::abs(mu_z.M() - z) < std::abs(el_z.M() - z)){
        z1 = mu_z;
        if (mu_charge[0] < 0) lep1_n = mu1;
        else lep1_n = mu2;
        z2 = el_z;
        if (el_charge[0] < 0) lep2_n = el1;
        else lep2_n = el2;
    } else{
        z1 = el_z;
        if (el_charge[0] < 0) lep1_n = el1;
        else lep1_n = el2;
        z2 = mu_z;
        if (mu_charge[0] < 0) lep2_n = mu1;
        else lep2_n = mu2;
    }
    
    beta_z1 = z1.BoostVector();
    beta_z2 = z2.BoostVector();

    lep1_n.Boost(-beta_z1);
    beta_lep1 = lep1_n.BoostVector();
    float theta1 = beta_lep1.Angle(beta_z1);

    lep2_n.Boost(-beta_z2);
    beta_lep2 = lep2_n.BoostVector();
    float theta2 = beta_lep2.Angle(beta_z2);

    costheta_vector[0] = TMath::Cos(theta1);
    costheta_vector[1] = TMath::Cos(theta2);

    return costheta_vector;
}

ROOT::VecOps::RVec<float> measure_phi(rvec el_pt, rvec el_eta, rvec el_phi, rvec el_mass, rvec el_charge,
                                      rvec mu_pt, rvec mu_eta, rvec mu_phi, rvec mu_mass, rvec mu_charge)
{
    TLorentzVector el1, el2, mu1, mu2, z1, z2, H, leading_lep, lep_same_charge, single_lep1;
	vector<TLorentzVector> lep(4), lep1(2), lep2(2);
    ROOT::VecOps::RVec<float> phi_vector(2), lep_charges(4);
    float pt_max, max_pos;
    TVector3 beta_H;

    mu1.SetPtEtaPhiM(mu_pt[0], mu_eta[0], mu_phi[0], mu_mass[0]);
    mu2.SetPtEtaPhiM(mu_pt[1], mu_eta[1], mu_phi[1], mu_mass[1]);
    el1.SetPtEtaPhiM(el_pt[0], el_eta[0], el_phi[0], el_mass[0]);
    el2.SetPtEtaPhiM(el_pt[1], el_eta[1], el_phi[1], el_mass[1]);

    lep = {mu1, mu2, el1, el2};
    lep_charges = {mu_charge[0], mu_charge[1], el_charge[0], el_charge[1]};

    pt_max = lep.at(0).Pt();
	max_pos = 0;
    for(int i=1; i<lep.size(); i++){ 
		if(lep.at(i).Pt()>pt_max) {
			pt_max = lep.at(i).Pt();
			max_pos = i;
		}
    }
    leading_lep = lep.at(max_pos);
    if (max_pos<2){
        if (lep_charges[0 + 2] == lep_charges[max_pos]) lep_same_charge = lep.at(0 + 2);
        else lep_same_charge = lep.at(1 + 2); 
    }
    else{
        if (lep_charges[0] == lep_charges[max_pos]) lep_same_charge = lep.at(0);
        else lep_same_charge = lep.at(1);         
    }

    auto mu_z = (mu1 + mu2);
    auto el_z = (el1 + el2);
    if(std::abs(mu_z.M() - z) < std::abs(el_z.M() - z)){
        z1 = mu_z;
        z2 = el_z;
    } else{
        z1 = el_z;
        z2 = mu_z;
    }

    H = z1 + z2;
    beta_H = H.BoostVector();

    single_lep1 = leading_lep;
	leading_lep.Boost(-beta_H);
    single_lep1.Boost(-beta_H);
	lep_same_charge.Boost(-beta_H);
	z1.Boost(-beta_H);

	phi_vector[0] = lep_same_charge.DeltaPhi(single_lep1);
	phi_vector[1] = z1.DeltaPhi(leading_lep);

    return phi_vector;
}