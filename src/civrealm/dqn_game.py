# Copyright (C) 2023  The CivRealm project
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

from civrealm.freeciv.utils.freeciv_logging import fc_logger
from civrealm.agents import DQNAgent
from civrealm.configs import fc_args
import civrealm
import gymnasium
import os
import torch
import time
import signal
import sys

def signal_handler(sig, frame):
    """Maneja la señal de interrupción para guardar el modelo y salir"""
    print("\nInterrumpiendo entrenamiento. Guardando modelo antes de salir...")
    agent._save_model(force=True)
    print("Modelo guardado. Saliendo...")
    sys.exit(0)

def main(num_episodes=5, continuous_training=True):
    """
    Función principal que ejecuta un juego con el agente DQN que aprende progresivamente.
    
    Args:
        num_episodes: Número de episodios (juegos) a ejecutar
        continuous_training: Si es True, seguirá entrenando indefinidamente
    """
    global agent
    
    try:
        # Registrar el manejador de señales
        signal.signal(signal.SIGINT, signal_handler)
        
        # Verificar que PyTorch esté disponible
        if not torch.cuda.is_available():
            fc_logger.info("CUDA no disponible, usando CPU para entrenamiento")
        else:
            fc_logger.info(f"CUDA disponible, usando GPU: {torch.cuda.get_device_name(0)}")
        
        # Crear el directorio para guardar los modelos si no existe
        model_dir = os.path.join(os.getcwd(), "dqn_models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            fc_logger.info(f"Directorio creado para guardar modelos: {model_dir}")
        
        # Crear el agente DQN con parámetros ajustados
        agent = DQNAgent(
            lr=0.001,
            gamma=0.99,
            epsilon=0.7,
            epsilon_min=0.05,
            epsilon_decay=0.995,
            batch_size=32,
            model_dir=model_dir
        )

        # Bucle de episodios (juegos completos)
        episode = 0
        total_steps = 0
        
        while episode < num_episodes or continuous_training:
            episode += 1
            print(f"=== Iniciando episodio {episode} ===")
            
            # Crear y configurar el entorno de juego
            env = gymnasium.make('civrealm/FreecivBase-v0')
            
            try:
                # Inicializar el entorno con un tiempo de espera para que se establezca la conexión
                observations, info = None, None
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        observations, info = env.reset(client_port=fc_args['client_port'])
                        if observations is not None and info is not None:
                            break
                    except Exception as e:
                        retry_count += 1
                        fc_logger.warning(f"Intento {retry_count}: Error al inicializar el entorno: {e}")
                        time.sleep(2)  # Esperar antes de reintentar
                
                if observations is None or info is None:
                    raise Exception("No se pudo inicializar el entorno después de varios intentos")
                
                # Variables de control para el bucle principal
                done = False
                step = 0
                
                # Bucle principal del juego
                while not done:
                    try:
                        # El agente decide una acción basada en las observaciones actuales
                        action = agent.act(observations, info)
                        
                        # Ejecutar la acción y obtener el siguiente estado
                        next_observations, reward, terminated, truncated, next_info = env.step(action)
                        
                        # Procesar la transición
                        if action is not None:  # Solo procesar si se tomó una acción válida
                            agent.process_transition(
                                observations, action, next_observations, 
                                reward, terminated or truncated, info, next_info
                            )
                        
                        # Actualizar estado
                        observations = next_observations
                        info = next_info
                        
                        # Mostrar información sobre el progreso
                        print(
                            f'Episodio: {episode}, Step: {step}, Turn: {info.get("turn", "?")}, '
                            f'Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}, '
                            f'action: {action}, Epsilon: {agent.epsilon:.4f}'
                        )
                        
                        # Actualizar contadores
                        step += 1
                        total_steps += 1
                        
                        # Verificar fin del episodio
                        done = terminated or truncated
                        
                    except Exception as e:
                        fc_logger.error(f"Error durante el paso {step}: {repr(e)}")
                        break
                
                # Guardar el modelo al final del episodio
                if step > 0:  # Solo si se completó al menos un paso
                    agent._save_model()
                    
            except Exception as e:
                fc_logger.error(f"Error en el episodio {episode}: {repr(e)}")
            finally:
                # Asegurarse de cerrar el entorno
                env.close()
            
            # Esperar antes del siguiente episodio
            if continuous_training or episode < num_episodes:
                print(f"Finalizó el episodio {episode}. Esperando 5 segundos antes de iniciar el siguiente...")
                time.sleep(5)
        
        # Mensaje final
        print(f"¡El agente DQN ha completado {episode} episodios con un total de {total_steps} pasos!")
        print(f"El entrenamiento ha avanzado hasta el paso {agent.training_step}")
        print("El modelo se ha guardado para uso futuro.")
    
    except Exception as e:
        fc_logger.error(f"Error general en la ejecución: {repr(e)}")
        # Intentar guardar el modelo incluso en caso de error
        try:
            if 'agent' in globals():
                agent._save_model(force=True)
                print("Modelo guardado a pesar del error.")
        except:
            pass
        raise e

if __name__ == '__main__':
    # Por defecto, entrenar continuamente hasta que el usuario interrumpa
    main(num_episodes=999999, continuous_training=True)