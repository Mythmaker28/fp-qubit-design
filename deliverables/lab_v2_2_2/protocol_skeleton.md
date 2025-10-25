# Experimental Protocol Skeleton
## Fluorescence-based Ion Channel Screening

### Overview
- **Total candidates**: 12
- **Families represented**: 8
- **Replicates per candidate**: 6 (96-well) / 2 (24-well)
- **Expected duration**: 2-3 days

### Instrument Parameters

#### Microplate Reader Settings
- **Temperature**: 37°C (maintained)
- **Read mode**: Fluorescence intensity
- **Integration time**: 100-200 ms per well
- **Gain**: Auto or optimized per filter set
- **Number of flashes**: 10-20 per measurement

### Spectral Parameters by Family

#### ATP Family (2 candidates)

**ATP_133**
- Excitation: 488 nm (468-508 nm)
- Emission: 515 nm (495-535 nm)
- Filter set: Exc [468, 508], Em [495, 535]

**ATP_114**
- Excitation: 488 nm (468-508 nm)
- Emission: 515 nm (495-535 nm)
- Filter set: Exc [468, 508], Em [495, 535]

#### Calcium Family (3 candidates)

**Calcium_33**
- Excitation: 488 nm (468-508 nm)
- Emission: 510 nm (490-530 nm)
- Filter set: Exc [468, 508], Em [490, 530]

**Calcium_14**
- Excitation: 488 nm (468-508 nm)
- Emission: 510 nm (490-530 nm)
- Filter set: Exc [468, 508], Em [490, 530]

**Calcium_20**
- Excitation: 488 nm (468-508 nm)
- Emission: 510 nm (490-530 nm)
- Filter set: Exc [468, 508], Em [490, 530]

#### GABA Family (1 candidates)

**GABA_111**
- Excitation: 488 nm (468-508 nm)
- Emission: 515 nm (495-535 nm)
- Filter set: Exc [468, 508], Em [495, 535]

#### NADH/NAD+ Family (1 candidates)

**NADH/NAD+_78**
- Excitation: 420 nm (400-440 nm)
- Emission: 535 nm (515-555 nm)
- Filter set: Exc [400, 440], Em [515, 555]

#### NADPH/NADP+ Family (1 candidates)

**NADPH/NADP+_205**
- Excitation: 420 nm (400-440 nm)
- Emission: 516 nm (496-536 nm)
- Filter set: Exc [400, 440], Em [496, 536]

#### Orange Family (1 candidates)

**Orange_123**
- Excitation: 406 nm (386-426 nm)
- Emission: 526 nm (506-546 nm)
- Filter set: Exc [386, 426], Em [506, 546]

#### Redox Family (2 candidates)

**Redox_121**
- Excitation: 405 nm (385-425 nm)
- Emission: 516 nm (496-536 nm)
- Filter set: Exc [385, 425], Em [496, 536]

**Redox_135**
- Excitation: 405 nm (385-425 nm)
- Emission: 516 nm (496-536 nm)
- Filter set: Exc [385, 425], Em [496, 536]

#### cAMP Family (1 candidates)

**cAMP_104**
- Excitation: 488 nm (468-508 nm)
- Emission: 510 nm (490-530 nm)
- Filter set: Exc [468, 508], Em [490, 530]

### Experimental Procedure

#### Day 1: Plate Preparation
1. **Buffer preparation** (pH 7.4, 37°C)
   - HEPES buffer: 10 mM HEPES, 140 mM NaCl, 5 mM KCl, 1 mM MgCl₂, 1 mM CaCl₂
   - Adjust pH to 7.4 ± 0.1
   - Filter sterilize (0.22 μm)

2. **Cell seeding**
   - Seed cells at 2×10⁴ cells/well (96-well) or 5×10⁴ cells/well (24-well)
   - Incubate at 37°C, 5% CO₂ for 24-48 hours

3. **Dye loading**
   - Load fluorescent indicators according to manufacturer protocol
   - Incubate for 30-60 minutes at 37°C
   - Wash 2× with buffer

#### Day 2: Experimental Measurements
1. **Baseline measurement** (5-10 cycles)
   - Read fluorescence for 2-5 minutes to establish baseline
   - Record F₀ (baseline fluorescence)

2. **Stimulus application**
   - Add test compounds or controls
   - Monitor fluorescence for 10-20 cycles
   - Record F₁ (stimulated fluorescence)

3. **Recovery measurement** (5-10 cycles)
   - Wash with buffer
   - Monitor fluorescence recovery
   - Record F₂ (recovery fluorescence)

### Quality Control

#### Data Validation
- **Outlier detection**: Exclude wells with residuals > P90 threshold
- **Replicate consistency**: CV < 20% between replicates
- **Signal-to-noise ratio**: SNR > 3:1
- **Minimum replicates**: n ≥ 3 per condition

#### Controls
- **Positive controls**: Known activators (n=8 per plate)
- **Negative controls**: Vehicle only (n=16 per plate)
- **Blank wells**: Buffer only (n=16 per plate)

### Data Analysis

#### Calculations
- **ΔF/F₀**: (F₁ - F₀) / F₀ × 100
- **Recovery**: (F₂ - F₀) / (F₁ - F₀) × 100
- **EC₅₀**: Concentration for 50% maximal response
- **Hill coefficient**: Steepness of dose-response curve

#### Statistical Analysis
- **ANOVA**: Compare between groups
- **Dunnett's test**: Multiple comparisons vs control
- **Dose-response fitting**: 4-parameter logistic model

### Documentation Requirements

#### Experimental Log
- **Date and time**: Record all measurements
- **Operator**: Initials of person performing experiment
- **Instrument settings**: Gain, integration time, filters
- **Environmental conditions**: Temperature, humidity

#### Data Storage
- **Raw data**: Fluorescence values per well
- **Metadata**: Plate layout, candidate information
- **Analysis files**: Processed data and statistics
- **DOI/Provenance**: Reference to Atlas database

### Safety Considerations

- **Personal protective equipment**: Lab coat, gloves, safety glasses
- **Chemical handling**: Follow SDS for all compounds
- **Waste disposal**: Segregate chemical waste appropriately
- **Emergency procedures**: Know location of safety equipment

### Notes

- **Buffer optimization**: May require pH/temperature adjustment
- **Timing optimization**: Adjust cycle number based on kinetics
- **Filter optimization**: Verify spectral overlap with indicators
- **Automation**: Consider robotic liquid handling for high-throughput

