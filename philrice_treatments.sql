-- ═══════════════════════════════════════════════════════════
--  PhilRice Treatment Data for RiceGuard
--  Source: PhilRice (Philippine Rice Research Institute)
--  Insert these into tbl_treatment after you fill in actual
--  dosage, application, and cost from PhilRice guidelines
-- ═══════════════════════════════════════════════════════════

-- First, check your disease IDs:
-- SELECT * FROM tbl_disease;
-- Expected: 1 = Bacterial Leaf Blight, 2 = Leaf Blast, 4 = Sheath Blight

-- ── Bacterial Leaf Blight (disease_id = 1) ───────────────
INSERT INTO tbl_treatment (disease_id, treatment_type, pesticide_name, dosage, application_method, estimated_cost)
VALUES
(1, 'Chemical', 'Copper Hydroxide (Kocide)', 'Fill from PhilRice', 'Foliar spray at booting stage', 0),
(1, 'Chemical', 'Streptomycin Sulfate', 'Fill from PhilRice', 'Foliar spray', 0),
(1, 'Cultural', 'Resistant Varieties (NSIC Rc222, Rc160)', 'Plant resistant variety', 'Use certified seeds', 0),
(1, 'Cultural', 'Balanced Fertilization', 'Avoid excessive nitrogen', 'Split application of fertilizer', 0);

-- ── Leaf Blast (disease_id = 2) ──────────────────────────
INSERT INTO tbl_treatment (disease_id, treatment_type, pesticide_name, dosage, application_method, estimated_cost)
VALUES
(2, 'Chemical', 'Tricyclazole (Beam)', 'Fill from PhilRice', 'Foliar spray at 1st symptom', 0),
(2, 'Chemical', 'Isoprothiolane (Fuji-one)', 'Fill from PhilRice', 'Foliar spray', 0),
(2, 'Cultural', 'Resistant Varieties (NSIC Rc160, Rc238)', 'Plant resistant variety', 'Use certified seeds', 0),
(2, 'Cultural', 'Silicon Application', '40-60 kg SiO2/ha', 'Basal application', 0);

-- ── Sheath Blight (disease_id = 4) ──────────────────────
INSERT INTO tbl_treatment (disease_id, treatment_type, pesticide_name, dosage, application_method, estimated_cost)
VALUES
(4, 'Chemical', 'Validamycin (Validacin)', 'Fill from PhilRice', 'Foliar spray at tillering', 0),
(4, 'Chemical', 'Azoxystrobin (Amistar)', 'Fill from PhilRice', 'Foliar spray at booting', 0),
(4, 'Cultural', 'Proper Spacing', '20 x 20 cm or wider', 'Transplant with wider spacing', 0),
(4, 'Cultural', 'Water Management', 'Alternate wetting and drying', 'Drain field periodically', 0);
