import argparse
import os
import ROOT
import time

# Enable multi-threading
# ROOT.ROOT.EnableImplicitMT()

def preliminary_analysis(path_to_file):
    """first look at the input file
    this function is meant to explore only first hand characteristics of events
    """
    list_branches = fIn.GetColumnNames()

    # decide which variables to look first
    dictOfBranches = { i:list_branches[i] for i in range (0,len(list_branches))}
    list_In = input('Insert the variable numbers to look at, separated by a space'
            f'press return when you are done \n {dictOfBranches}:\n')
        
    #control input and retrieve the required branches
    list_str = list_In.split(" ")
    branches_in = []
    for i in list_str:
        if i.isdigit():
            branches_in.append(dictOfBranches[int(i)])
        else: print(f'Error! {i} is an invalid key!')
    print(branches_in)
    return branches_in
    

def plot(data, x_label, filename):
    """For now it's just a simple plot of unprocessed data
    """
    #General Canvas Settings
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetTextFont(42)
    c_histo = ROOT.TCanvas("c_histo","",400,400)
    
    #Set and Draw histogram for data points
    histo_data = data
    histo_data.SetLineStyle(0)
    histo_data.SetLineWidth(1)
    histo_data.SetLineColor(1)
    histo_data.SetMarkerStyle(20)
    histo_data.SetMarkerColor(ROOT.kBlack)
    histo_data.SetMarkerSize(0.5)
    histo_data.Draw()

    #Add Legend
    legend = ROOT.TLegend(0.60, 0.70, 0.8, 0.9)
    legend.SetFillColor(0)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.05)
    legend.AddEntry(histo_data, "Data","D")
    legend.Draw()

    #Save plot
    c_histo.SaveAs(filename)

if __name__ == '__main__':
    
    #Insert input file from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='path to the input file')
    args = parser.parse_args()
    try:
        fIn = ROOT.RDataFrame("Events",args.infile)

        #Perform a preliminar analysis
        branch_names = preliminary_analysis(fIn)
        for branch in branch_names:
            hist1 = fIn.Histo1D(branch)
            hist1.Draw()
            plot(hist1.GetValue(), branch, branch+'.pdf')
            time.sleep(1)
        
        #3D reconstruction of some fundamental variables
        pv_3d = fIn.Define("PV_3d","sqrt(PV_x*PV_x + PV_y*PV_y + PV_z+PV_z)")
        muon_3d = fIn.Define("Muon_3d","sqrt(Muon_dxy*Muon_dxy + Muon_dz*Muon_dz)")
        el_3d = fIn.Define("El_3d","sqrt(Electron_dxy*Electron_dxy + Electron_dz*Electron_dz)")

        h_pv_3d = pv_3d.Histo1D("PV_3d")
        h_muon_3d = muon_3d.Histo1D("Muon_3d")
        h_el_3d = el_3d.Histo1D("El_3d")

        plot(h_pv_3d.GetValue(),"PV_3d","PV_3d.pdf")
        plot(h_muon_3d.GetValue(),"Muon_3d","Muon_3d.pdf")
        plot(h_el_3d.GetValue(),"El_3d","El_3d.pdf")


    except OSError as e:
        print(f'Cannot read the file!\n{e}') 


