#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <TAxis.h>
#include <TFile.h>
#include <TH1.h>
#include <TEfficiency.h>

enum Cuts {
    in_p = 0,
    in_t = 1,
    tf_p = 2,
    tf_t = 3,
    l1_p = 4,
    l1_t = 5,
    main_phys = 6,
    main_unb = 7
};

struct EffVars {
    uint8_t mask;
    float rig_gen, rig;
    float rig_beta, rig_inl1, rig_inn;
    void set_mask_bits(int i) {
        mask |= (1 << i);
    }
    bool check_mask_bits(int i) {
        return mask & (1 << i);
    }
};

enum Det {inn, tf, l1, tr, acc};
inline constexpr std::array<const char*, 5> DetNames = {"in", "tf", "l1", "tr", "acc"};
inline constexpr std::array<Det, 5> DetList = {inn, tf, l1, tr, acc};

struct McFileInfo {
    Long64_t nentries;
};

class McRW {
    public:
    McRW(const char* filename) {
        this->file = TFile::Open(filename, "READ");
        std::cout << "File opened: " << this->file->GetName() << std::endl;
    };

    void set_rbins(const std::vector<double>& rbins, const std::vector<double>& rbins_acc);

    
    void load_tree();
    
    void fill_sums(const std::vector<double>& weights = {});
    
    const McFileInfo& get_mc_file_info() {return mc_file_info;};
    
    const std::array<std::pair<std::vector<double>, std::vector<double>>, 5>& get_sums_pass() {return sums_pass;};
    
    const std::array<std::pair<std::vector<double>, std::vector<double>>, 5>& get_sums_total() {return sums_total;};

    private:
    TFile* file;
    TAxis* rbins, *rbins_acc;

    McFileInfo mc_file_info;

    std::vector<float> rig_gen {};
    std::vector<uint8_t> mask {};
    std::array<std::vector<uint16_t>, 5> rig_bin_idx {};

    std::array<std::pair<std::vector<double>, std::vector<double>>, 5> sums_pass;
    std::array<std::pair<std::vector<double>, std::vector<double>>, 5> sums_total;
};