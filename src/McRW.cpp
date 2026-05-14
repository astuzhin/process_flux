#include "McRW.hpp"
#include <TTree.h>
#include <array>

inline bool check_mask_bits(uint8_t mask, int i) {
    return mask & (1 << i);
}

void McRW::set_rbins(const std::vector<double>& rbins, const std::vector<double>& rbins_acc) {
    this->rbins = new TAxis(rbins.size() - 1, rbins.data());
    this->rbins_acc = new TAxis(rbins_acc.size() - 1, rbins_acc.data());
    for (auto det : DetList) {
        if (det == Det::acc) {
            sums_pass[det].first.resize(this->rbins_acc->GetNbins() + 2, 0.0);
            sums_pass[det].second.resize(this->rbins_acc->GetNbins() + 2, 0.0);
            sums_total[det].first.resize(this->rbins_acc->GetNbins() + 2, 0.0);
            sums_total[det].second.resize(this->rbins_acc->GetNbins() + 2, 0.0);
        } else {
            sums_pass[det].first.resize(this->rbins->GetNbins() + 2, 0.0);
            sums_pass[det].second.resize(this->rbins->GetNbins() + 2, 0.0);
            sums_total[det].first.resize(this->rbins->GetNbins() + 2, 0.0);
            sums_total[det].second.resize(this->rbins->GetNbins() + 2, 0.0);
        }
    }
}

void McRW::load_tree() {
    std::cout << "Loading tree..." << std::endl;
    auto* tree = (TTree*)this->file->Get("eff_tree");
    EffVars vars;
    tree->SetBranchAddress("rig_gen", &vars.rig_gen);
    tree->SetBranchAddress("rig", &vars.rig);
    tree->SetBranchAddress("rig_beta", &vars.rig_beta);
    tree->SetBranchAddress("rig_inl1", &vars.rig_inl1);
    tree->SetBranchAddress("rig_inn", &vars.rig_inn);
    tree->SetBranchAddress("mask", &vars.mask);
    mc_file_info.nentries = tree->GetEntries();
    for (int i = 0; i < tree->GetEntries(); i++) {
        tree->GetEntry(i);

        float rig_in = vars.rig_beta < 5.0 ? vars.rig_beta : vars.rig_gen;
        float rig_tf = vars.rig_inl1;
        float rig_l1 = vars.rig_inn;
        float rig_tr = vars.rig;

        uint16_t bin_in = static_cast<uint16_t>(rbins->FindBin(rig_in));
        uint16_t bin_tf = static_cast<uint16_t>(rbins->FindBin(rig_tf));
        uint16_t bin_l1 = static_cast<uint16_t>(rbins->FindBin(rig_l1));
        uint16_t bin_tr = static_cast<uint16_t>(rbins->FindBin(rig_tr));
        uint16_t bin_acc = static_cast<uint16_t>(rbins_acc->FindBin(rig_tr));

        // std::cout << Form("rig_in: %f, rig_tf: %f, rig_l1: %f, rig_tr: %f, rig_gen: %f", rig_in, rig_tf, rig_l1, rig_tr, vars.rig_gen) << std::endl;
        // std::cout << Form("bin_in: %d, bin_tf: %d, bin_l1: %d, bin_tr: %d, bin_gen: %d", bin_in, bin_tf, bin_l1, bin_tr, bin_gen) << std::endl << std::endl;

        rig_gen.push_back(vars.rig_gen);
        mask.push_back(vars.mask);
        rig_bin_idx[Det::inn].push_back(bin_in);
        rig_bin_idx[Det::tf].push_back(bin_tf);
        rig_bin_idx[Det::l1].push_back(bin_l1);
        rig_bin_idx[Det::tr].push_back(bin_tr);
        rig_bin_idx[Det::acc].push_back(bin_acc);
    }
}

void McRW::fill_sums(const std::vector<double>& weights) {
    bool use_weights = !weights.empty();
    // if (use_weights) {
        // std::cout << "Filling sums with weights ..." << std::endl;
    // } else {
        // std::cout << "Filling sums without weights ..." << std::endl;
    // }
    for (auto det : DetList) {
        std::fill(sums_pass[det].first.begin(), sums_pass[det].first.end(), 0.0);
        std::fill(sums_pass[det].second.begin(), sums_pass[det].second.end(), 0.0);
        std::fill(sums_total[det].first.begin(), sums_total[det].first.end(), 0.0);
        std::fill(sums_total[det].second.begin(), sums_total[det].second.end(), 0.0);
    }

    auto accum = [&](Det det, bool is_total, bool is_pass, int i, double w, double w2) {
        if (!is_total) return;
        const int bin = rig_bin_idx[det][i];
        sums_total[det].first[bin] += w;
        sums_total[det].second[bin] += w2;
        if (is_pass) {
            sums_pass[det].first[bin] += w;
            sums_pass[det].second[bin] += w2;
        }
    };
    
    for (int i = 0; i < mask.size(); i++) {
        double w = use_weights ? weights[i] : 1.0;
        double w2 = w * w;
        
        accum(Det::l1, check_mask_bits(mask[i], Cuts::l1_t), check_mask_bits(mask[i], Cuts::l1_p), i, w, w2);
        accum(Det::tf, check_mask_bits(mask[i], Cuts::tf_t), check_mask_bits(mask[i], Cuts::tf_p), i, w, w2);
        accum(Det::inn, check_mask_bits(mask[i], Cuts::in_t), check_mask_bits(mask[i], Cuts::in_p), i, w, w2);
        bool pass_phys = check_mask_bits(mask[i], Cuts::main_phys);
        bool pass_unb = check_mask_bits(mask[i], Cuts::main_unb);
        accum(Det::tr, (pass_phys || pass_unb), pass_phys, i, w, w2);

        if (pass_phys) {
            const int bin = rig_bin_idx[Det::acc][i];
            sums_pass[Det::acc].first[bin] += w;
            sums_pass[Det::acc].second[bin] += w2;
        }
    }
}