from os import path
from os.path import abspath
from pathlib import Path

from midas.util.runtime_config import RuntimeConfig
from pysimmods.generator.chplpgsystemsim.presets import chp_preset

# data paths
ROOT_PATH = Path(abspath(__file__)).parent.parent
data_path = ROOT_PATH / 'data'
WIND_DATA = data_path / 'wind_speed_2020.csv'
POWER_CURVE_DATA_FILE = ROOT_PATH / 'power_coeff_curve.csv'
PV_DATA = data_path / 'pv_10kw.csv'
ROOT_PATH_midas = Path(abspath(__file__)).parent
data_path_midas = ROOT_PATH_midas / 'data'
MIDAS_DATA = data_path_midas / 'midas_data' / RuntimeConfig().data["smart_nord"][0]["name"]

# scenario configuration
STEP_SIZE = 900
NUM_SIMULATIONS = 200
# Maximum Wind at this time
START = '2022-07-23 14:00:00'
START_FORMAT_CSV = '23.07.2022 14:00:00'
END = '2022-07-23 15:00:00'
SIMULATION_HOURS = 6
SIMULATION_HOURS_IN_RESOLUTION = int(SIMULATION_HOURS * 4)
NUMBER_OF_SCHEDULES_PER_AGENT = 10

NUMBER_OF_WIND_AGENTS = 10
NUMBER_OF_PV_AGENTS = 10
NUMBER_OF_CHPS = 10
NUMBER_OF_BATTERIES = 10
NUMBER_OF_HOUSEHOLDS = 10

NUMBER_OF_AGENTS_TOTAL = NUMBER_OF_WIND_AGENTS + NUMBER_OF_CHPS + NUMBER_OF_PV_AGENTS + NUMBER_OF_BATTERIES + NUMBER_OF_HOUSEHOLDS

NEGOTIATION_TIMEOUT = 200

db_file = 'results' + START + '.hdf5'
# 0: No Failure
# 5: Asset failure
# 6: Communication failure
# 7. Communication delays
# 8. Compromised process data
# 9. No communication from Aggregator Agent to neighborhood grid
# 10. No communication from neighborhood grid to Aggregator agent
ATTACK_SCENARIO = 11
# set to: '2' for attacks 0-4 and 8 (partition id is 2)
# set to: 'generation_agent_1' for attacks 5, 6, 7 (agent id)
# set to "aggregator_agent" for 9, 10
MANIPULATED_AGENT_ID = 'aggregator_agent'

SCHEDULE_PERCENTAGES = [0.2, 0.4, 0.6, 0.8, 1.0]

# UNIT PARAMETERS
PARAMS_BATT = {
    'cap_kwh': .05,
    'p_charge_max_kw': .1,
    'p_discharge_max_kw': .1,
    'soc_min_percent': 15,
    'eta_pc': [-2.109566, 0.403556, 97.110770],
}
INITS_BATT = {
    'soc_percent': 50
}

POSSIBLE_KW = [7, 14, 200, 400]
CHP_PARAMS, CHP_INITS = chp_preset(200)  # It is not working for 150 kW

# weather information
T_AIR = "WeatherCurrent__0___t_air_deg_celsius"
WIND = "WeatherCurrent__0___wind_v_m_per_s"
WIND_DIR = "WeatherCurrent__0___wind_dir_deg"
PRESSURE = "WeatherCurrent__0___air_pressure_hpa"
WD_PATH = path.abspath(
    path.join(__file__, "..", "data", "weather-time-series.csv")
)

BH = "WeatherCurrent__0___bh_w_per_m2"
DH = "WeatherCurrent__0___dh_w_per_m2"

INITIAL_TARGET_PARAMS = {
    'current_start_date': START}
