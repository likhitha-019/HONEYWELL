# process_overview.py
import streamlit as st

def show_process_overview():
    """
    Displays the detailed F&B process overview, quality definitions, and references.
    This fulfills Hackathon deliverables 1, 2, and 4.
    """
    st.header("🍞 Industrial Bakery Process Overview & Theory")

    st.subheader("1. Process Steps for Bread Production")
    st.markdown("""
    The manufacturing of baked goods is a precise biochemical process. Below are the key stages and their critical control parameters:

    **1. Mixing & Ingredient Addition**
    - **Purpose:** Hydrates dry ingredients and develops gluten structure.
    - **Equipment:** Horizontal or Spiral Mixer
    - **Key Control Parameters:**
        - `mixing_time_min`: Under-mixing results in poor gluten development; over-mixing causes dough breakdown.
        - `mixing_speed_rpm`: Impacts mechanical energy input and dough temperature rise.
        - `ingredient quantities (flour_kg, water_kg, etc.)`: Precise ratios are critical. Water temperature is often used to control final dough temp.

    **2. Fermentation (Bulk Fermentation)**
    - **Purpose:** Yeast consumes sugars, producing CO₂ (for volume) and organic compounds (for flavor).
    - **Equipment:** Fermentation Cabinet or Room
    - **Key Control Parameters:**
        - `fermentation_time_min`: Directly impacts flavor profile and gas production.
        - `fermentation_temp_c`: Ideal range is 27-30°C. Higher temperatures accelerate activity.

    **3. Proofing (Final Proof)**
    - **Purpose:** Final rise of the shaped dough before baking.
    - **Equipment:** Proofing Cabinet
    - **Key Control Parameters:**
        - `proofing_time_min`: Must be optimized. Under-proofed bread is dense; over-proofed bread collapses.
        - `proofing_temp_c`: Typically 35-40°C with high humidity (~80% RH) to prevent skin formation.
        - `proofing_rh_percent`: (Simulated in our model) Humidity control is critical.

    **4. Baking**
    - **Purpose:** Sets the structure, creates crust, develops flavor via Maillard reaction, and ensures food safety.
    - **Equipment:** Multi-zone Tunnel Oven
    - **Key Control Parameters:**
        - `baking_time_min`: Determines moisture evaporation and final weight.
        - `baking_temp_c`, `oven_zone1_temp_c`, `zone2_temp_c`, `zone3_temp_c`: Modern ovens have different temperature zones.
            - **Zone 1 (High Heat):** Causes "oven spring" - rapid expansion.
            - **Zone 2 (Moderate Heat):** Sets the crumb structure.
            - **Zone 3 (Lower Heat):** Develops color and flavor.

    **5. Cooling**
    - **Purpose:** Allows moisture to redistribute and the structure to set before packaging.
    - **Equipment:** Cooling Conveyor
    - **Key Control Parameters:**
        - `cooling_time_min`: Prevents condensation in packaging, which causes mold.
        - `cooling_temp_c`: Ambient or controlled temperature.
    """)

    st.subheader("2. Definition of Final Product Quality")
    st.markdown("""
    Product quality is quantitatively defined based on measurable attributes:

    - **Core Temperature (`core_temp_c`):** Must reach a minimum of **92-99°C** to ensure starch gelatinization is complete and the product is safe for consumption. *Measured by a thermocouple probe.*

    - **Weight Deviation (`weight_deviation`):** Measures evaporation loss during baking. An ideal batch has **< 2%** deviation from the target input weight. *Measured by a checkweigher.*

    - **Color Score (`color_score`):** Quantifies crust browning from the Maillard reaction. Scored from 0-100, with >90 being the target for premium products. *Measured by a machine vision system or colorimeter.*

    - **Overall Quality Score (`quality_score`):** A composite, weighted index (0-100) derived from the above parameters. **Scores below 60 indicate a process anomaly.**
    """)

    st.subheader("3. Justification of Data Streams & Parameters")
    st.markdown("""
    The selected process parameters are standard data streams collected by PLCs and SCADA systems in modern bakeries:
    - **Raw Material Quantities:** Tracked via load cells in ingredient silos and hoppers. Deviations indicate dosing errors or supplier inconsistency.
    - **Time & Speed Parameters:** Controlled by the recipe management system (e.g., SAP ME).
    - **Temperature Parameters:** Measured by RTD sensors and thermocouples throughout the process. Critical for controlling biochemical reactions.
    - **Quality Data:** Captured by inline sensors (checkweighers, vision systems) at the end of the line.
    """)

    st.subheader("4. References")
    st.markdown("""
    This process overview and parameter selection is based on industry standards and academic literature:
    - [1] Cauvain, S.P. & Young, L.S. (2007). *Technology of Breadmaking* (2nd ed.). Springer.
    - [2] [AIBI (International Association of Plant Bakeries) Best Practices Guidelines](https://www.aibi.eu/)
    - [3] [Baking Process Technology](https://www.bakingbusiness.com/)
    - [4] [UCI Machine Learning Repository](https://archive.ics.uci.edu/): Datasets for predictive maintenance and process analysis.
    - [5] Honeywell Process Solutions. (2021). *White Paper: Improving Efficiency in Food & Beverage Manufacturing*.
    """)

# This allows the file to be run on its own for testing
if __name__ == "__main__":
    show_process_overview()