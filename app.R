
# ============================================================================
# AMI-CS Early Hemodynamic Deterioration Risk Calculator
# V20-lite XGBoost Model — Shiny Application
# ============================================================================
# This app accompanies the manuscript:
#   "Machine Learning-Based Prediction of Early Hemodynamic Deterioration
#    in Acute Myocardial Infarction Complicated by Cardiogenic Shock"
#   Submitted to Circulation: Cardiovascular Quality and Outcomes
# ============================================================================


install.packages(shinythemes)

library(shiny)
library(xgboost)
library(shinythemes)

# ── Load model artifacts ──────────────────────────────────────────────────
xgb_model      <- readRDS("v20_xgb_lite.rds")
v20_lite_all   <- readRDS("v20_lite_all.rds")
v20_lite_meds  <- readRDS("v20_lite_medians.rds")
model_features <- readRDS("model_features.rds")
train_medians  <- readRDS("train_medians.rds")

# ── Helper: safe numeric ─────────────────────────────────────────────────
sn <- function(x) suppressWarnings(as.numeric(x))
safe_div <- function(x, y) ifelse(is.finite(x) & is.finite(y) & y != 0, x / y, NA_real_)

# ── V20 feature computation (mirrors v20_add from training pipeline) ─────
compute_v20_features <- function(input_list) {
  # Extract raw inputs
  map_0h   <- sn(input_list$map_0h)
  map_6h   <- sn(input_list$map_6h)
  map_12h  <- sn(input_list$map_12h)
  map_18h  <- sn(input_list$map_18h)
  map_24h  <- sn(input_list$map_24h)
  lact_0h  <- sn(input_list$lactate_0h)
  lact_12h <- sn(input_list$lactate_12h)
  lact_24h <- sn(input_list$lactate_24h)
  creat_0h <- sn(input_list$creatinine_0h)
  creat_24h <- sn(input_list$creatinine_24h)
  uo_24h   <- sn(input_list$total_urine_output_24h)
  weight   <- sn(input_list$weight_0h)
  vaso     <- as.integer(sn(input_list$any_vasoactive_0_6h) > 0)
  
  # MAP matrix
  map_vec <- c(map_0h, map_6h, map_12h, map_18h, map_24h)
  map_valid <- map_vec[is.finite(map_vec)]
  
  # Lactate matrix
  lact_vec <- c(lact_0h, lact_12h, lact_24h)
  lact_valid <- lact_vec[is.finite(lact_vec)]
  
  # ── V20-lite 8 core features ──
  uo_ml_kg_h <- if (is.finite(weight) && weight > 0 && is.finite(uo_24h)) {
    uo_24h / (weight * 24)
  } else NA_real_
  
  v20_uo_deficit <- if (is.finite(uo_ml_kg_h)) max(0, 0.5 - uo_ml_kg_h) else NA_real_
  v20_oliguria   <- as.integer(is.finite(uo_ml_kg_h) && uo_ml_kg_h < 0.5)
  v20_creat_ratio <- safe_div(creat_24h, creat_0h)
  
  v20_map_area_below65 <- if (length(map_valid) > 0) {
    sum(pmax(0, 65 - map_valid))
  } else 0
  
  map_below65_count <- sum(map_vec < 65, na.rm = TRUE)
  v20_persist_hypo <- as.integer(map_below65_count >= 2)
  v20_vaso_bin     <- vaso
  
  # Shock intensity score
  lact_hi4_prop <- if (length(lact_valid) > 0) {
    sum(lact_valid > 4) / length(lact_valid)
  } else 0
  
  v20_shock_intensity <- (
    (if (is.finite(lact_hi4_prop)) lact_hi4_prop * 2 else 0) +
    (if (is.finite(map_below65_count)) pmin(map_below65_count / 5, 1) * 2 else 0) +
    (if (is.finite(v20_creat_ratio)) pmax(0, v20_creat_ratio - 1) else 0) +
    (if (is.finite(v20_uo_deficit)) v20_uo_deficit * 2 else 0)
  )
  
  v20_miss_lact <- sum(!is.finite(lact_vec))
  
  # ── Build full feature vector ──
  # Start with all base model features set to training medians
  feat_vec <- train_medians[model_features]
  
  # Overwrite with user-provided values
  raw_map <- list(
    age = input_list$age,
    sofa_score_0h = input_list$sofa_score_0h,
    cci_score = input_list$cci_score,
    map_0h_median = map_0h, sbp_0h_median = input_list$sbp_0h,
    hr_0h_median = input_list$hr_0h,
    lactate_0h_median = lact_0h, creatinine_0h_median = creat_0h,
    temp_0h_median_celsius = input_list$temp_0h,
    glucose_0h_median = input_list$glucose_0h,
    hemoglobin_0h_median = input_list$hemoglobin_0h,
    ph_0h_median = input_list$ph_0h,
    map_0h = map_0h, map_6h = map_6h, map_12h = map_12h,
    map_18h = map_18h, map_24h = map_24h,
    hr_0h = input_list$hr_0h,
    spo2_0h = input_list$spo2_0h,
    rr_0h = input_list$rr_0h,
    lactate_0h = lact_0h, lactate_12h = lact_12h, lactate_24h = lact_24h,
    creatinine_0h = creat_0h, creatinine_24h = creat_24h,
    total_urine_output_24h = uo_24h,
    gender_Male = input_list$gender_Male,
    vasoactive_Yes = vaso,
    shock_index_0h = safe_div(sn(input_list$hr_0h), sn(input_list$sbp_0h)),
    uo_ml_kg_h = uo_ml_kg_h,
    oliguria_flag = as.integer(is.finite(uo_ml_kg_h) && uo_ml_kg_h < 0.3),
    creatinine_ratio_24h = v20_creat_ratio,
    creatinine_change_0_24h = if (is.finite(creat_24h) && is.finite(creat_0h)) creat_24h - creat_0h else NA_real_,
    lactate_clearance_rate = if (is.finite(lact_0h) && lact_0h > 0 && is.finite(lact_24h)) (lact_0h - lact_24h) / lact_0h else NA_real_,
    lactate_per_map_ratio_0h = safe_div(lact_0h, map_0h),
    lactate_per_map_ratio_24h = safe_div(lact_24h, map_24h),
    map_change_0_24h = if (is.finite(map_24h) && is.finite(map_0h)) map_24h - map_0h else NA_real_
  )
  
  for (nm in names(raw_map)) {
    val <- sn(raw_map[[nm]])
    if (nm %in% names(feat_vec) && is.finite(val)) {
      feat_vec[nm] <- val
    }
  }
  
  # Add V20 features
  v20_feats <- c(
    v20_uo_deficit = ifelse(is.finite(v20_uo_deficit), v20_uo_deficit, 0),
    v20_oliguria = v20_oliguria,
    v20_creat_ratio = ifelse(is.finite(v20_creat_ratio), v20_creat_ratio, 1),
    v20_map_area_below65 = v20_map_area_below65,
    v20_persist_hypo = v20_persist_hypo,
    v20_vaso_bin = v20_vaso_bin,
    v20_shock_intensity = v20_shock_intensity,
    v20_miss_lact = v20_miss_lact
  )
  
  all_feats <- c(feat_vec, v20_feats)
  
  # Select only features used by the model, in correct order
  out <- numeric(length(v20_lite_all))
  names(out) <- v20_lite_all
  for (f in v20_lite_all) {
    if (f %in% names(all_feats) && is.finite(all_feats[f])) {
      out[f] <- all_feats[f]
    } else if (f %in% names(v20_lite_meds)) {
      out[f] <- v20_lite_meds[f]
    } else {
      out[f] <- 0
    }
  }
  
  matrix(out, nrow = 1, dimnames = list(NULL, v20_lite_all))
}

# ── Risk interpretation ──────────────────────────────────────────────────
interpret_risk <- function(prob) {
  if (prob < 0.15) {
    list(level = "Low", color = "#27AE60",
         text = "Low risk of hemodynamic deterioration in the next 24-48 hours. Standard monitoring recommended.")
  } else if (prob < 0.35) {
    list(level = "Moderate", color = "#F39C12",
         text = "Moderate risk. Consider enhanced hemodynamic monitoring, serial lactate measurements, and reassessment of vasopressor/inotrope therapy.")
  } else if (prob < 0.55) {
    list(level = "High", color = "#E67E22",
         text = "High risk. Recommend intensive monitoring, early cardiology/cardiac surgery consultation, and evaluation for mechanical circulatory support readiness.")
  } else {
    list(level = "Very High", color = "#E74C3C",
         text = "Very high risk of deterioration. Urgent reassessment of treatment strategy. Consider early MCS (IABP/Impella/ECMO) if not already initiated. Multidisciplinary shock team activation recommended.")
  }
}

# ══════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════
ui <- fluidPage(
  theme = shinytheme("flatly"),
  
  tags$head(tags$style(HTML("
    .risk-box { padding: 20px; border-radius: 10px; margin: 15px 0;
                text-align: center; color: white; font-size: 18px; }
    .input-section { background: #f8f9fa; padding: 15px; border-radius: 8px;
                     margin-bottom: 15px; }
    .input-section h4 { color: #2C3E50; margin-top: 0; border-bottom: 2px solid #3498DB;
                        padding-bottom: 8px; }
    .prob-display { font-size: 48px; font-weight: bold; }
    .disclaimer { font-size: 11px; color: #7f8c8d; margin-top: 20px;
                  padding: 10px; background: #fdf2e9; border-radius: 5px; }
    .v20-box { background: #eaf2f8; padding: 12px; border-radius: 8px;
               margin-top: 10px; font-size: 13px; }
  "))),
  
  titlePanel(
    div(
      h2("AMI-CS Early Hemodynamic Deterioration Risk Calculator",
         style = "margin-bottom: 2px;"),
      h5("V20-lite XGBoost Model — 24-48h Prediction",
         style = "color: #7f8c8d; margin-top: 0;")
    )
  ),
  
  sidebarLayout(
    sidebarPanel(
      width = 5,
      
      # Demographics
      div(class = "input-section",
        h4(icon("user"), "Demographics"),
        fluidRow(
          column(4, numericInput("age", "Age (years)", value = 68, min = 18, max = 110)),
          column(4, selectInput("gender", "Sex", choices = c("Male" = 1, "Female" = 0), selected = 1)),
          column(4, numericInput("weight", "Weight (kg)", value = 80, min = 30, max = 250))
        )
      ),
      
      # Admission Vitals (0h)
      div(class = "input-section",
        h4(icon("heartbeat"), "Admission Vitals (0-6h)"),
        fluidRow(
          column(3, numericInput("map_0h", "MAP (mmHg)", value = 65, min = 20, max = 200)),
          column(3, numericInput("sbp_0h", "SBP (mmHg)", value = 90, min = 40, max = 300)),
          column(3, numericInput("hr_0h", "HR (bpm)", value = 95, min = 20, max = 250)),
          column(3, numericInput("spo2_0h", "SpO2 (%)", value = 95, min = 50, max = 100))
        ),
        fluidRow(
          column(4, numericInput("rr_0h", "RR (bpm)", value = 20, min = 4, max = 60)),
          column(4, numericInput("temp_0h", "Temp (°C)", value = 36.5, min = 30, max = 42, step = 0.1)),
          column(4, selectInput("vaso_0_6h", "Vasopressors (0-6h)?",
                                choices = c("Yes" = 1, "No" = 0), selected = 1))
        )
      ),
      
      # MAP Trajectory
      div(class = "input-section",
        h4(icon("chart-line"), "MAP Trajectory (6h intervals)"),
        helpText("Enter MAP values at each time point. Leave NA if unavailable."),
        fluidRow(
          column(3, numericInput("map_6h", "6h MAP", value = NA, min = 20, max = 200)),
          column(3, numericInput("map_12h", "12h MAP", value = NA, min = 20, max = 200)),
          column(3, numericInput("map_18h", "18h MAP", value = NA, min = 20, max = 200)),
          column(3, numericInput("map_24h", "24h MAP", value = 70, min = 20, max = 200))
        )
      ),
      
      # Laboratories
      div(class = "input-section",
        h4(icon("flask"), "Laboratory Values"),
        fluidRow(
          column(4, numericInput("lactate_0h", "Lactate 0h (mmol/L)", value = 3.5, min = 0.1, max = 30, step = 0.1)),
          column(4, numericInput("lactate_12h", "Lactate 12h", value = NA, min = 0.1, max = 30, step = 0.1)),
          column(4, numericInput("lactate_24h", "Lactate 24h", value = NA, min = 0.1, max = 30, step = 0.1))
        ),
        fluidRow(
          column(4, numericInput("creat_0h", "Creatinine 0h (mg/dL)", value = 1.2, min = 0.1, max = 20, step = 0.1)),
          column(4, numericInput("creat_24h", "Creatinine 24h", value = NA, min = 0.1, max = 20, step = 0.1)),
          column(4, numericInput("glucose_0h", "Glucose (mg/dL)", value = 180, min = 20, max = 1000))
        ),
        fluidRow(
          column(3, numericInput("hemoglobin_0h", "Hb (g/dL)", value = 11.0, min = 2, max = 25, step = 0.1)),
          column(3, numericInput("ph_0h", "pH", value = 7.35, min = 6.8, max = 7.8, step = 0.01)),
          column(3, numericInput("sofa_0h", "SOFA score", value = 8, min = 0, max = 24)),
          column(3, numericInput("cci", "CCI", value = 5, min = 0, max = 15))
        )
      ),
      
      # Urine Output
      div(class = "input-section",
        h4(icon("tint"), "Fluid Balance (0-24h)"),
        numericInput("urine_24h", "Total urine output, 0-24h (mL)", value = 800, min = 0, max = 10000, step = 50)
      ),
      
      # Calculate button
      actionButton("calc_btn", "Calculate Risk",
                   class = "btn-primary btn-lg btn-block",
                   icon = icon("calculator"),
                   style = "margin-top: 10px; font-size: 18px;")
    ),
    
    mainPanel(
      width = 7,
      
      # Risk output
      uiOutput("risk_output"),
      
      # V20-lite feature display
      div(class = "v20-box",
        h4("V20-lite Composite Feature Values", style = "margin-top: 0;"),
        tableOutput("v20_features_table")
      ),
      
      # Model information
      br(),
      tabsetPanel(
        tabPanel("About the Model",
          br(),
          h4("V20-lite: 8-Feature XGBoost Model"),
          p("This calculator implements the V20-lite model, a parsimonious
             XGBoost-based classifier using 8 SHAP-derived composite features
             for predicting early (24-48h) hemodynamic deterioration in patients
             with acute myocardial infarction complicated by cardiogenic shock."),
          tags$table(class = "table table-striped table-sm",
            tags$thead(tags$tr(
              tags$th("#"), tags$th("Feature"), tags$th("Description")
            )),
            tags$tbody(
              tags$tr(tags$td("1"), tags$td("UO Deficit"), tags$td("max(0, 0.5 - UO mL/kg/h)")),
              tags$tr(tags$td("2"), tags$td("Oliguria"), tags$td("UO < 0.5 mL/kg/h (binary)")),
              tags$tr(tags$td("3"), tags$td("Creatinine Ratio"), tags$td("Cr 24h / Cr baseline")),
              tags$tr(tags$td("4"), tags$td("MAP Area Below 65"), tags$td("Cumulative MAP deficit < 65 mmHg")),
              tags$tr(tags$td("5"), tags$td("Persistent Hypotension"), tags$td("MAP < 65 in ≥2/4 time bins")),
              tags$tr(tags$td("6"), tags$td("Vasopressor Use"), tags$td("Any vasoactive agent 0-6h")),
              tags$tr(tags$td("7"), tags$td("Shock Intensity"), tags$td("Weighted composite of lactate, MAP, Cr, UO")),
              tags$tr(tags$td("8"), tags$td("Lactate Missingness"), tags$td("Missing lactate count (0/12/24h)"))
            )
          ),
          h4("Performance"),
          tags$ul(
            tags$li("Internal validation (MIMIC-IV, N=327): AUC = 0.815 (0.763–0.867)"),
            tags$li("External validation (eICU-CRD, N=653): AUC = 0.756 (0.716–0.795)"),
            tags$li("Brier score: 0.156 (internal), 0.186 (external, after recalibration)")
          )
        ),
        tabPanel("Outcome Definition",
          br(),
          h4("24-48h Early Hemodynamic Deterioration"),
          p("The composite endpoint includes ANY of the following occurring
             between 24 and 48 hours after ICU admission:"),
          tags$ol(
            tags$li("Mean arterial pressure < 60 mmHg"),
            tags$li("Lactate > 10% increase from baseline"),
            tags$li("Lactate ≥ 4 mmol/L"),
            tags$li("Urine output < 0.3 mL/kg/h"),
            tags$li("Creatinine > 50% increase from baseline"),
            tags$li("New vasopressor initiation"),
            tags$li("Vasopressor escalation"),
            tags$li("Death within 24-48h")
          )
        ),
        tabPanel("References",
          br(),
          p("This tool accompanies the following manuscript:"),
          p(tags$em("Interpretable machine learning for dynamic risk prediction of hemodynamic deterioration in acute myocardial infarction with cardiogenic shock: development, external validation, and clinical decision support deployment using the MIMIC-IV and eICU-CRD databases")),
          p("Submitted to: BMC Cardiovascular Disorders"),
          br(),
          p(tags$strong("Data sources:")),
          tags$ul(
            tags$li("Development: MIMIC-IV v3.1 (N = 1,633; 2008-2019)"),
            tags$li("External validation: eICU-CRD v2.0 (N = 653; 2014-2015)")
          )
        )
      ),
      
      # Disclaimer
      div(class = "disclaimer",
        tags$strong("⚠ Clinical Disclaimer: "),
        "This tool is intended for research purposes only and has not been
         prospectively validated for clinical decision-making. Predictions
         should be interpreted in the context of the full clinical picture
         by qualified healthcare professionals. This model was developed
         using retrospective data from US-based ICU databases and may not
         generalize to all clinical settings. Site-specific recalibration
         is recommended before deployment."
      )
    )
  )
)

# ══════════════════════════════════════════════════════════════════════════
# SERVER
# ══════════════════════════════════════════════════════════════════════════
server <- function(input, output, session) {
  
  # Reactive: compute prediction on button click
  prediction <- eventReactive(input$calc_btn, {
    
    input_list <- list(
      age = input$age,
      gender_Male = as.integer(input$gender),
      weight_0h = input$weight,
      map_0h = input$map_0h,
      sbp_0h = input$sbp_0h,
      hr_0h = input$hr_0h,
      spo2_0h = input$spo2_0h,
      rr_0h = input$rr_0h,
      temp_0h = input$temp_0h,
      any_vasoactive_0_6h = as.integer(input$vaso_0_6h),
      map_6h = input$map_6h,
      map_12h = input$map_12h,
      map_18h = input$map_18h,
      map_24h = input$map_24h,
      lactate_0h = input$lactate_0h,
      lactate_12h = input$lactate_12h,
      lactate_24h = input$lactate_24h,
      creatinine_0h = input$creat_0h,
      creatinine_24h = input$creat_24h,
      glucose_0h = input$glucose_0h,
      hemoglobin_0h = input$hemoglobin_0h,
      ph_0h = input$ph_0h,
      sofa_score_0h = input$sofa_0h,
      cci_score = input$cci,
      total_urine_output_24h = input$urine_24h
    )
    
    # Compute feature matrix
    X_new <- compute_v20_features(input_list)
    
    # Predict
    dmat <- xgb.DMatrix(X_new)
    prob <- predict(xgb_model, dmat)
    
    # V20-lite feature values for display
    v20_names <- c("v20_uo_deficit", "v20_oliguria", "v20_creat_ratio",
                   "v20_map_area_below65", "v20_persist_hypo", "v20_vaso_bin",
                   "v20_shock_intensity", "v20_miss_lact")
    v20_vals <- X_new[1, intersect(v20_names, colnames(X_new))]
    
    list(probability = prob, v20_values = v20_vals, input_list = input_list)
  })
  
  # Render risk output
  output$risk_output <- renderUI({
    req(prediction())
    prob <- prediction()$probability
    risk <- interpret_risk(prob)
    
    tagList(
      div(class = "risk-box", style = paste0("background-color:", risk$color, ";"),
        p(class = "prob-display", sprintf("%.1f%%", prob * 100)),
        h3(paste("Risk Level:", risk$level)),
        p(risk$text)
      ),
      
      # Progress bar
      div(style = "margin: 10px 20px;",
        div(style = "display: flex; justify-content: space-between; font-size: 12px; color: #7f8c8d;",
          span("0%"), span("25%"), span("50%"), span("75%"), span("100%")
        ),
        div(style = paste0(
          "background: linear-gradient(to right, #27AE60 0%, #F39C12 30%, #E67E22 55%, #E74C3C 100%);",
          "height: 12px; border-radius: 6px; position: relative;"
        ),
          div(style = paste0(
            "position: absolute; left: ", prob * 100, "%;",
            "top: -4px; width: 20px; height: 20px; background: white;",
            "border: 3px solid ", risk$color, "; border-radius: 50%;",
            "transform: translateX(-50%);"
          ))
        )
      )
    )
  })
  
  # Render V20 feature table
  output$v20_features_table <- renderTable({
    req(prediction())
    v20_vals <- prediction()$v20_values
    
    display_names <- c(
      "v20_uo_deficit"       = "Urine Output Deficit",
      "v20_oliguria"         = "Oliguria Indicator",
      "v20_creat_ratio"      = "24h Creatinine Ratio",
      "v20_map_area_below65" = "MAP Area Below 65 mmHg",
      "v20_persist_hypo"     = "Persistent Hypotension",
      "v20_vaso_bin"         = "Vasopressor Use (0-6h)",
      "v20_shock_intensity"  = "Shock Intensity Score",
      "v20_miss_lact"        = "Lactate Missingness Count"
    )
    
    data.frame(
      Feature = display_names[names(v20_vals)],
      Value = sprintf("%.3f", as.numeric(v20_vals)),
      stringsAsFactors = FALSE
    )
  }, striped = TRUE, hover = TRUE, spacing = "s", width = "100%")
}

# ── Launch ────────────────────────────────────────────────────────────────
shinyApp(ui = ui, server = server)


