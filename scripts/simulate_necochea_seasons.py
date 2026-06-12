import traci
import argparse
import random
import sys

# Definiciones de los edges (calles) de entrada en SUMO para Necochea
# Asumiremos nombres genéricos, pero deben coincidir con la red SUMO real (.net.xml)
EDGE_AVENUE_TO_BEACH = "avenue_in_to_beach"
EDGE_AVENUE_TO_DOWNTOWN = "avenue_in_to_downtown"
EDGE_CROSS_1 = "cross_in_1"
EDGE_CROSS_2 = "cross_in_2"

def get_probabilities(season, time_of_day_s):
    """
    Retorna las probabilidades de generación de vehículos (to_beach, to_downtown, cross)
    basado en la estación y la hora del día.
    """
    if season == "invierno":
        # Invierno: tráfico bajo constante. La gente no va masivamente a la playa.
        # Noches súper vacías.
        if 0 <= time_of_day_s < 6 * 3600: # Madrugada
            return 0.001, 0.001, 0.0001
        else:
            return 0.02, 0.02, 0.005

    elif season == "verano":
        # Verano: Picos direccionales
        # 11:00 a 14:00 (39600 a 50400) -> Flujo masivo hacia la costa
        if 11 * 3600 <= time_of_day_s <= 14 * 3600:
            return 0.50, 0.10, 0.05
        # 18:30 a 21:30 (66600 a 77400) -> Flujo masivo hacia el centro (regreso)
        elif 18.5 * 3600 <= time_of_day_s <= 21.5 * 3600:
            return 0.10, 0.50, 0.05
        # Noche de verano (movimiento hacia el centro/boliches)
        elif 22 * 3600 <= time_of_day_s <= 24 * 3600 or 0 <= time_of_day_s <= 3 * 3600:
            return 0.05, 0.15, 0.02
        else:
            # Día regular de verano
            return 0.20, 0.20, 0.10
            
    return 0.0, 0.0, 0.0

def generate_traffic(season, current_time_s, veh_counter):
    """
    Inyecta vehículos en SUMO mediante TraCI dependiendo de las probabilidades calculadas.
    """
    prob_beach, prob_down, prob_cross = get_probabilities(season, current_time_s)
    
    # Inyectar hacia la playa
    if random.random() < prob_beach:
        veh_id = f"veh_beach_{veh_counter[0]}"
        try:
            traci.vehicle.add(vehID=veh_id, routeID="route_to_beach")
            veh_counter[0] += 1
        except traci.exceptions.TraCIException:
            pass # Ignorar si la ruta no existe (mock environment)

    # Inyectar hacia el centro
    if random.random() < prob_down:
        veh_id = f"veh_down_{veh_counter[1]}"
        try:
            traci.vehicle.add(vehID=veh_id, routeID="route_to_downtown")
            veh_counter[1] += 1
        except traci.exceptions.TraCIException:
            pass
            
    # Inyectar en calles transversales
    if random.random() < prob_cross:
        veh_id = f"veh_cross_{veh_counter[2]}"
        try:
            traci.vehicle.add(vehID=veh_id, routeID="route_cross")
            veh_counter[2] += 1
        except traci.exceptions.TraCIException:
            pass

def run_simulation(season, net_file, start_hour=8.0):
    """
    Ejecuta la simulación con la estacionalidad dada.
    """
    sumo_cmd = ["sumo-gui", "-n", net_file] # Usa sumo-gui para visualizar
    try:
        traci.start(sumo_cmd)
    except traci.exceptions.FatalTraCIError as e:
        print(f"Error al iniciar SUMO: {e}")
        print("Asegúrese de tener una red válida (-n necochea.net.xml) y SUMO instalado.")
        sys.exit(1)
        
    print(f"--- Iniciando Simulación de {season.capitalize()} ---")
    
    # Offset simulado para arrancar a una hora específica
    time_offset_s = start_hour * 3600 
    veh_counter = [0, 0, 0] # beach, down, cross
    
    step = 0
    while traci.simulation.getMinExpectedNumber() >= 0:
        traci.simulationStep()
        
        # Calcular hora del día simulada (loop de 24h)
        current_time_s = (traci.simulation.getTime() + time_offset_s) % 86400
        
        generate_traffic(season, current_time_s, veh_counter)
        
        step += 1
        # Simulamos 24 horas (86400 pasos)
        if step > 86400:
            break
            
    traci.close()
    print("--- Simulación Finalizada ---")
    print(f"Vehículos generados: Playa={veh_counter[0]}, Centro={veh_counter[1]}, Transversales={veh_counter[2]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulador de Tráfico Estacional - Necochea")
    parser.add_argument("--season", choices=["invierno", "verano"], required=True, help="Estación del año a simular")
    parser.add_argument("--net", default="necochea.net.xml", help="Archivo de red de SUMO (.net.xml)")
    parser.add_argument("--start-hour", type=float, default=10.0, help="Hora de inicio simulada (0.0 a 24.0)")
    args = parser.parse_args()
    
    run_simulation(args.season, args.net, args.start_hour)
