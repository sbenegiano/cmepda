// mass 4l combined datasets 2011 and 2012
//
// Original Author:
//         Created:  Fri December 11, 2017 by  N.Z. Jomhari   
//                   with contributions from   A. Geiser
//                                             A. Anuar
//
// ***********************************************************
// use this root macro to generate the Higgs->4L mass plot   *
// from the histograms produced from the original data files *   
// ***********************************************************

#include "TROOT.h" 
#include "iostream"
#include <iomanip>
#include "TH1.h"
#include "TCanvas.h"
#include "TPad.h"
#include "stdlib.h"
#include "TStyle.h"
#include "TH1D.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TLegend.h"
#include "TFile.h"
#include "string.h"
#include "TChain.h"
#include "TTree.h"
#include "TGaxis.h"
#include "TLatex.h"
#include "TBufferFile.h"
#include "TLorentzVector.h"
#include "THStack.h"
#include <math.h>

void M4Lnormdatall() {

  TH1::SetDefaultSumw2(kTRUE);
  
  // Input file directory
  string inDir = "./";  
  // Output file directory
  string outDir = "./";

  // Name of the input file for MC
  string inFileZZ4mu12 = "ZZ4mu12.root";
  string inFileZZ4e12 = "ZZ4e12.root";
  string inFileZZ2mu2e12 = "ZZ2mu2e12.root";
  string inFileZZ4mu11 = "ZZ4mu11.root";
  string inFileZZ4e11 = "ZZ4e11.root";
  string inFileZZ2mu2e11 = "ZZ2mu2e11.root";

  string inFileHZZ12 = "HZZ12.root";
  string inFileHZZ11 = "HZZ11.root";

  string inFileTTBar12 = "TTBar12.root";
  string inFileTTBar11 = "TTBar11.root";

  string inFileDY5012 = "DY50TuneZ12.root";
  string inFileDY5011 = "DY50TuneZ11.root";
  string inFileDY1012 = "DY1012.root";
  string inFileDY1011 = "DY1011.root";

  // Name of input file for data
  string inFileDouMu12 = "DoubleMu12.root";
  string inFileDouMu11 = "DoubleMu11.root";
  string inFileDouE12 = "DoubleE12.root";
  string inFileDouE11 = "DoubleE11.root";

  // Name of your output file
  // Preferably save the output file in pdf
  // Open the output file using: $xdg-open mass4l_combine.pdf 
  string outFile = "mass4l_combine_user.pdf";

  // Luminosity 2012 and 2011
  Double_t lumi12 = 11580.;
  Double_t lumi11 = 2330.;

  // MC cross section 
  Double_t xsecZZ412 = 0.077;
  Double_t xsecZZ2mu2e12 = 0.18;
  Double_t xsecZZ411 = 0.067;
  Double_t xsecZZ2mu2e11 = 0.15;

  Double_t xsecTTBar12 = 200.;
  Double_t xsecTTBar11 = 177.31;

  Double_t xsecDY5012 = 2955.;
  Double_t xsecDY1012 = 10.742;
  Double_t xsecDY5011 = 2475.;
  Double_t xsecDY1011 = 9507.;
  
  // Scale factor 
  Double_t sfZZ = 1.386;
  Double_t sfTTBar11 = 0.11;
  Double_t sfDY = 1.12;

  // For Higgs case, we use scaled cross section
  Double_t scalexsecHZZ12 = 0.0065;
  Double_t scalexsecHZZ11 = 0.0057;

  // No. of event
  Int_t nevtZZ4mu12 = 1499064;
  Int_t nevtZZ4e12 = 1499093;
  Int_t nevtZZ2mu2e12 = 1497445;
  Int_t nevtHZZ12 = 299973;
  Int_t nevtTTBar12 = 6423106;
  Int_t nevtDY5012 = 29426492;
  Int_t nevtDY1012 = 6462290;
  
  Int_t nevtZZ4mu11 = 1447136;
  Int_t nevtZZ4e11 = 1493308;
  Int_t nevtZZ2mu2e11 = 1479879;
  Int_t nevtHZZ11 = 299683;
  Int_t nevtTTBar11 = 9771205;
  Int_t nevtDY5011 = 36408225;
  Int_t nevtDY1011 = 39909640;

  // Title of the plot
  string title = "#sqrt{s} = 7 TeV, L = 2.3 fb^{-1}, #sqrt{s} = 8 TeV, L = 11.6 fb^{-1}";
  
  // y axis maximum
  Double_t yMax = 30.;
  
  ////////////////////// HISTOGRAM FOR ZZ PROCESS 2012 ///////////////////////////

  // ZZ -> 4mu
  TFile *f2 = new TFile((inDir + inFileZZ4mu12).c_str());
  TH1D *ZZto4mu12 = (TH1D*) f2->Get("demo/mass4mu_8TeV_low")->Clone();
  ZZto4mu12->Scale((lumi12 * xsecZZ412 * sfZZ) / nevtZZ4mu12);
  //(data lumi * xsec * scale factor) / no.of event b4 any cut

  // ZZ -> 4e
  f2 = TFile::Open((inDir + inFileZZ4e12).c_str());
  TH1D *ZZto4e12 = (TH1D*) f2->Get("demo/mass4e_8TeV_low")->Clone();
  ZZto4e12->Scale((lumi12 * xsecZZ412 * sfZZ) / nevtZZ4e12); 

  // ZZ -> 2mu2e
  f2 = TFile::Open((inDir + inFileZZ2mu2e12).c_str());
  TH1D *ZZto2mu2e12 = (TH1D*) f2->Get("demo/mass2mu2e_8TeV_low")->Clone();
  ZZto2mu2e12->Scale((lumi12 * xsecZZ2mu2e12 * sfZZ) / nevtZZ2mu2e12); 
  
  ////////////////////// HISTOGRAM FOR ZZ PROCESS 2011 ///////////////////////////

  // ZZ -> 4mu
  f2 = TFile::Open((inDir + inFileZZ4mu11).c_str());
  TH1D *ZZto4mu11 = (TH1D*) f2->Get("demo/mass4mu_8TeV_low")->Clone();
  ZZto4mu11->Scale((lumi11 * xsecZZ411 * sfZZ) / nevtZZ4mu11); 

  // ZZ -> 4e
  f2 = TFile::Open((inDir + inFileZZ4e11).c_str());
  TH1D *ZZto4e11 = (TH1D*) f2->Get("demo/mass4e_8TeV_low")->Clone();
  ZZto4e11->Scale((lumi11 * xsecZZ411 * sfZZ) / nevtZZ4e11); 

  // ZZ -> 2mu2e
  f2 = TFile::Open((inDir + inFileZZ2mu2e11).c_str());
  TH1D *ZZto2mu2e11 = (TH1D*) f2->Get("demo/mass2mu2e_8TeV_low")->Clone();
  ZZto2mu2e11->Scale((lumi11 * xsecZZ2mu2e11 * sfZZ) / nevtZZ2mu2e11); 
  
  ///////////////////////////// COMBINE ZZ PROCESS ///////////////////////////////

  TH1D *ZZto4l = (TH1D*) ZZto4mu12->Clone();
  ZZto4l->Add(ZZto4e12);
  ZZto4l->Add(ZZto2mu2e12);
  ZZto4l->Add(ZZto4mu11);
  ZZto4l->Add(ZZto4e11);
  ZZto4l->Add(ZZto2mu2e11);
  
  gStyle->SetOptStat(0);

  ZZto4l->SetLineColor(kBlack);
  ZZto4l->SetFillColor(kAzure -9);
  ZZto4l->SetLineWidth(2);
  ZZto4l->SetLineStyle(1);
  
  //////////////////// HISTOGRAM FOR HIGGS PROCESS 2012 //////////////////////////

  // H -> ZZ -> 4mu
  f2 = TFile::Open((inDir + inFileHZZ12).c_str());
  TH1D *HZZto4mu12 = (TH1D*) f2->Get("demo/mass4mu_8TeV_low")->Clone();
  HZZto4mu12->Scale((lumi12 * scalexsecHZZ12) / nevtHZZ12); 
  //(data lumi * scaled cross section) / no.of event b4 any cut

  // H -> ZZ -> 4e
  f2 = TFile::Open((inDir + inFileHZZ12).c_str());
  TH1D *HZZto4e12 = (TH1D*) f2->Get("demo/mass4e_8TeV_low")->Clone();
  HZZto4e12->Scale((lumi12 * scalexsecHZZ12) / nevtHZZ12); 

  // H -> ZZ -> 2mu2e
  f2 = TFile::Open((inDir + inFileHZZ12).c_str());
  TH1D *HZZto2mu2e12 = (TH1D*) f2->Get("demo/mass2mu2e_8TeV_low")->Clone();
  HZZto2mu2e12->Scale((lumi12 * scalexsecHZZ12) / nevtHZZ12);
  
  //////////////////// HISTOGRAM FOR HIGGS PROCESS 2011 //////////////////////////

  // H -> ZZ -> 4mu
  f2 = TFile::Open((inDir + inFileHZZ11).c_str());
  TH1D *HZZto4mu11 = (TH1D*) f2->Get("demo/mass4mu_8TeV_low")->Clone();
  HZZto4mu11->Scale((lumi11 * scalexsecHZZ11) / nevtHZZ11); 

  // H -> ZZ -> 4e
  f2 = TFile::Open((inDir + inFileHZZ11).c_str());
  TH1D *HZZto4e11 = (TH1D*) f2->Get("demo/mass4e_8TeV_low")->Clone();
  HZZto4e11->Scale((lumi11 * scalexsecHZZ11) / nevtHZZ11); 

  // H -> ZZ -> 2mu2e
  f2 = TFile::Open((inDir + inFileHZZ11).c_str());
  TH1D *HZZto2mu2e11 = (TH1D*) f2->Get("demo/mass2mu2e_8TeV_low")->Clone();
  HZZto2mu2e11->Scale((lumi11 * scalexsecHZZ11) / nevtHZZ11);
  
  //////////////////////////// COMBINE HZZ PROCESS ///////////////////////////////
  
  TH1D *HZZto4l = (TH1D*) HZZto4mu12->Clone();
  HZZto4l->Add(HZZto4e12);
  HZZto4l->Add(HZZto2mu2e12);
  HZZto4l->Add(HZZto4mu11);
  HZZto4l->Add(HZZto4e11);
  HZZto4l->Add(HZZto2mu2e11);

  HZZto4l->SetLineColor(kRed);
  HZZto4l->SetLineWidth(2);
  HZZto4l->SetLineStyle(1);
  
  ///////////////////// HISTOGRAM FOR TTBAR PROCESS 2012 /////////////////////////
  
  // TTBar -> 4mu
  f2 = TFile::Open((inDir + inFileTTBar12).c_str());
  TH1D *TTBarto4mu12 = (TH1D*) f2->Get("demo/mass4mu_8TeV_low")->Clone();
  TTBarto4mu12->Scale((lumi12 * xsecTTBar12) / nevtTTBar12); 

  // TTBar -> 4e
  f2 = TFile::Open((inDir + inFileTTBar12).c_str());
  TH1D *TTBarto4e12 = (TH1D*) f2->Get("demo/mass4e_8TeV_low")->Clone();
  TTBarto4e12->Scale((lumi12 * xsecTTBar12) / nevtTTBar12);

  // TTBar -> 2mu2e
  f2 = TFile::Open((inDir + inFileTTBar12).c_str());
  TH1D *TTBarto2mu2e12 = (TH1D*) f2->Get("demo/mass2mu2e_8TeV_low")->Clone();
  TTBarto2mu2e12->Scale((lumi12 * xsecTTBar12) / nevtTTBar12);
  
  ///////////////////// HISTOGRAM FOR TTBAR PROCESS 2011 /////////////////////////

  // TTBar -> 4mu
  f2 = TFile::Open((inDir + inFileTTBar11).c_str());
  TH1D *TTBarto4mu11 = (TH1D*) f2->Get("demo/mass4mu_8TeV_low")->Clone();
  TTBarto4mu11->Scale((lumi11 * xsecTTBar11 * sfTTBar11) / nevtTTBar11); 
  //(data lumi * xsec * scale factor) / no.of event b4 any cut

  // TTBar -> 4e
  f2 = TFile::Open((inDir + inFileTTBar11).c_str());
  TH1D *TTBarto4e11 = (TH1D*) f2->Get("demo/mass4e_8TeV_low")->Clone();
  TTBarto4e11->Scale((lumi11 * xsecTTBar11 * sfTTBar11) / nevtTTBar11); 

  // TTBar -> 2mu2e
  f2 = TFile::Open((inDir + inFileTTBar11).c_str());
  TH1D *TTBarto2mu2e11 = (TH1D*) f2->Get("demo/mass2mu2e_8TeV_low")->Clone();
  TTBarto2mu2e11->Scale((lumi11 * xsecTTBar11 * sfTTBar11) / nevtTTBar11);
  
  /////////////////////////// COMBINE TTBAR PROCESS //////////////////////////////

  TH1D *TTBarto4l = (TH1D*) TTBarto4mu12->Clone();
  TTBarto4l->Add(TTBarto4e12);
  TTBarto4l->Add(TTBarto2mu2e12);
  TTBarto4l->Add(TTBarto4mu11);
  TTBarto4l->Add(TTBarto4e11);
  TTBarto4l->Add(TTBarto2mu2e11);
  
  TTBarto4l->SetLineColor(kBlack);
  TTBarto4l->SetFillColor(kGray);
  TTBarto4l->SetLineWidth(2);
  TTBarto4l->SetLineStyle(1);
  
  ///////////////////// HISTOGRAM FOR DRLY PROCESS 2012 //////////////////////////

  // DY 50 -> 4mu
  f2 = TFile::Open((inDir + inFileDY5012).c_str());
  TH1D *DY504mu12 = (TH1D*) f2->Get("demo/mass4mu_8TeV_low")->Clone();
  DY504mu12->Scale((lumi12 * xsecDY5012 * sfDY) / nevtDY5012);

  // DY 50 -> 4e
  f2 = TFile::Open((inDir + inFileDY5012).c_str());
  TH1D *DY504e12 = (TH1D*) f2->Get("demo/mass4e_8TeV_low")->Clone();
  DY504e12->Scale((lumi12 * xsecDY5012 * sfDY) / nevtDY5012); 

  // DY 50 -> 2mu2e
  f2 = TFile::Open((inDir + inFileDY5012).c_str());
  TH1D *DY502mu2e12 = (TH1D*) f2->Get("demo/mass2mu2e_8TeV_low")->Clone();
  DY502mu2e12->Scale((lumi12 * xsecDY5012 * sfDY) / nevtDY5012); 

  // DY 10 -> 4mu
  f2 = TFile::Open((inDir + inFileDY1012).c_str());
  TH1D *DY104mu12 = (TH1D*) f2->Get("demo/mass4mu_8TeV_low")->Clone();
  DY104mu12->Scale((lumi12 * xsecDY1012 * sfDY) / nevtDY1012); 

  // DY 10 -> 4e
  f2 = TFile::Open((inDir + inFileDY1012).c_str());
  TH1D *DY104e12 = (TH1D*) f2->Get("demo/mass4e_8TeV_low")->Clone();
  DY104e12->Scale((lumi12 * xsecDY1012 * sfDY) / nevtDY1012);

  // DY 10 -> 2mu2e
  f2 = TFile::Open((inDir + inFileDY1012).c_str());
  TH1D *DY102mu2e12 = (TH1D*) f2->Get("demo/mass2mu2e_8TeV_low")->Clone();
  DY102mu2e12->Scale((lumi12 * xsecDY1012 * sfDY) / nevtDY1012);
  
  ///////////////////// HISTOGRAM FOR DRLY PROCESS 2011 //////////////////////////

  // DY 50 -> 4mu
  f2 = TFile::Open((inDir + inFileDY5011).c_str());
  TH1D *DY504mu11 = (TH1D*) f2->Get("demo/mass4mu_8TeV_low")->Clone();
  DY504mu11->Scale((lumi11 * xsecDY5011 * sfDY) / nevtDY5011);

  // DY 50 -> 4e
  f2 = TFile::Open((inDir + inFileDY5011).c_str());
  TH1D *DY504e11 = (TH1D*) f2->Get("demo/mass4e_8TeV_low")->Clone();
  DY504e11->Scale((lumi11 * xsecDY5011 * sfDY) / nevtDY5011); 

  // DY 50 -> 2mu2e
  f2 = TFile::Open((inDir + inFileDY5011).c_str());
  TH1D *DY502mu2e11 = (TH1D*) f2->Get("demo/mass2mu2e_8TeV_low")->Clone();
  DY502mu2e11->Scale((lumi11 * xsecDY5011 * sfDY) / nevtDY5011); 

  // DY 10 -> 4mu
  f2 = TFile::Open((inDir + inFileDY1011).c_str());
  TH1D *DY104mu11 = (TH1D*) f2->Get("demo/mass4mu_8TeV_low")->Clone();
  DY104mu11->Scale((lumi11 * xsecDY1011 * sfDY) / nevtDY1011); 

  // DY 10 -> 4e
  f2 = TFile::Open((inDir + inFileDY1011).c_str());
  TH1D *DY104e11 = (TH1D*) f2->Get("demo/mass4e_8TeV_low")->Clone();
  DY104e11->Scale((lumi11 * xsecDY1011 * sfDY) / nevtDY1011); 

  // DY 10 -> 2mu2e
  f2 = TFile::Open((inDir + inFileDY1011).c_str());
  TH1D *DY102mu2e11 = (TH1D*) f2->Get("demo/mass2mu2e_8TeV_low")->Clone();
  DY102mu2e11->Scale((lumi11 * xsecDY1011 * sfDY) / nevtDY1011); 
  
  /////////////////////////// COMBINE PROCESS DRLY ///////////////////////////////

  TH1D *DYto4l = (TH1D*) DY504mu12->Clone();
  DYto4l->Add(DY504e12);
  DYto4l->Add(DY502mu2e12);
  DYto4l->Add(DY104mu12);
  DYto4l->Add(DY104e12);
  DYto4l->Add(DY102mu2e12);

  DYto4l->Add(DY504mu11);
  DYto4l->Add(DY504e11);
  DYto4l->Add(DY502mu2e11);
  DYto4l->Add(DY104mu11);
  DYto4l->Add(DY104e11);
  DYto4l->Add(DY102mu2e11);

  DYto4l->SetLineColor(kBlack);
  DYto4l->SetFillColor(kGreen -5);
  DYto4l->SetLineWidth(2);
  DYto4l->SetLineStyle(1);
  
  /////////////////////// HISTOGRAM FOR DATA!!!!! 2012 ///////////////////////////
  
  // Double Mu (4mu)
  TFile *f3 = new TFile((inDir + inFileDouMu12).c_str());
  TH1D *DoubleMu4mu12 = (TH1D*) f3->Get("demo/mass4mu_8TeV_low")->Clone();

  // Double Mu (2mu2e)
  f3 = TFile::Open((inDir + inFileDouMu12).c_str());
  TH1D *DoubleMu2mu2e12 = (TH1D*) f3->Get("demo/mass2mu2e_8TeV_low")->Clone();

  // Double E (4e)
  f3 = TFile::Open((inDir + inFileDouE12).c_str());
  TH1D *DoubleE4e12 = (TH1D*) f3->Get("demo/mass4e_8TeV_low")->Clone();
  
  /////////////////////// HISTOGRAM FOR DATA!!!!! 2011 ///////////////////////////

  // Double Mu (4mu)
  f3 = TFile::Open((inDir + inFileDouMu11).c_str());
  TH1D *DoubleMu4mu11 = (TH1D*) f3->Get("demo/mass4mu_8TeV_low")->Clone();

  // Double Mu (2mu2e)  
  f3 = TFile::Open((inDir + inFileDouMu11).c_str());
  TH1D *DoubleMu2mu2e11 = (TH1D*) f3->Get("demo/mass2mu2e_8TeV_low")->Clone();

  // Double E (4e)
  f3 = TFile::Open((inDir + inFileDouE11).c_str());
  TH1D *DoubleE4e11 = (TH1D*) f3->Get("demo/mass4e_8TeV_low")->Clone();

  ///////////////////////// COMBINE DATA!!!!!!!!!!!! /////////////////////////////
  
  TH1D *Dat114l = (TH1D*) DoubleMu4mu12->Clone();
  Dat114l->Add(DoubleMu2mu2e12);
  Dat114l->Add(DoubleE4e12);
  Dat114l->Add(DoubleMu4mu11);
  Dat114l->Add(DoubleMu2mu2e11);
  Dat114l->Add(DoubleE4e11);

  Dat114l->SetMarkerColor(kBlack);
  Dat114l->SetMarkerStyle(20);
  Dat114l->SetMarkerSize(1.7);
  Dat114l->SetLineColor(kBlack);
  Dat114l->SetLineWidth(1);

  THStack *mcomb = new THStack("mcomb", " ");
  mcomb->Add(DYto4l);
  mcomb->Add(TTBarto4l);
  mcomb->Add(ZZto4l);
  mcomb->Add(HZZto4l);
  
  TLegend *leg = new TLegend(.62, .70, .82, .88);
  leg->SetFillColor(0);
  leg->SetBorderSize(0);
  leg->SetTextFont(42);
  leg->SetTextSize(0.038);

  leg->AddEntry(Dat114l, "Data", "PE1");
  leg->AddEntry(DYto4l, "Z/#gamma* + X", "f");
  leg->AddEntry(TTBarto4l, "TTBar", "f");
  leg->AddEntry(ZZto4l, "ZZ -> 4l", "f");
  leg->AddEntry(HZZto4l, "m_{H} = 125 GeV", "f");

  TCanvas *c1 = new TCanvas("c1", "", 200, 10, 1000, 1000);

  Dat114l->Draw("PE1");
  mcomb->Draw("hist");
  Dat114l->Draw("PE1 same");

  mcomb->GetXaxis()->SetTitle("m_{4l} (GeV)");
  mcomb->GetYaxis()->SetTitle("Events / 3 GeV");
  mcomb->GetYaxis()->SetTitleOffset(1.2);
  mcomb->SetMaximum(yMax);

  TLatex *txt1 = new TLatex();
  txt1->DrawTextNDC(0.15,0.84,"CMS Open Data");
  TLatex *txt2 = new TLatex();
  txt2->SetTextSize(0.03);
  txt2->SetNDC();
  txt2->SetTextFont(42);
  txt2->DrawLatex(0.14, 0.91, title.c_str());

  leg->Draw();

  c1->SaveAs(outFile.c_str());
  //  c1->Close();
  //  gROOT->ProcessLine(".q");
   
}
