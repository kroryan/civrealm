import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import glob
import json
import pickle
import time
import traceback
import signal
import sys

from civrealm.agents.base_agent import BaseAgent
from civrealm.agents.controller_agent import ControllerAgent
from civrealm.freeciv.utils.freeciv_logging import fc_logger

# Experiencia almacenada para el entrenamiento
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'expert_action'])

class ReplayMemory:
    """Memoria de experiencias para entrenamiento"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, state, action, next_state, reward, done, expert_action):
        """Guarda una transición en la memoria"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward, done, expert_action)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Obtiene un lote aleatorio de la memoria"""
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self):
        return len(self.memory)

class DQNetwork(nn.Module):
    """Red neuronal unificada para DQN"""
    def __init__(self, input_size=150, output_size=100):
        super(DQNetwork, self).__init__()
        # Capa de entrada con normalización
        self.input_layer = nn.Linear(input_size, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        
        # Capas ocultas 
        self.hidden1 = nn.Linear(256, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        
        # Capa de salida
        self.output_layer = nn.Linear(128, output_size)
        
        # Dropout para regularización
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Asegurarse de que x tiene el formato correcto
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Añadir dimensión de batch

        try:
            # Pasar por las capas con activaciones y normalización
            x = F.relu(self.batch_norm1(self.input_layer(x)))
            x = self.dropout(x)
            x = F.relu(self.batch_norm2(self.hidden1(x)))
            x = self.dropout(x)
            
            # Capa de salida (sin activación, son Q-values)
            return self.output_layer(x)
        except Exception as e:
            fc_logger.error(f"Error en forward pass: {e}")
            return torch.zeros((x.size(0), self.output_layer.out_features), device=x.device)

class DQNAgent(BaseAgent):
    def __init__(self, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                 batch_size=64, model_dir="dqn_models", max_turns=500, imitation_weight=0.5, checkpoint_frequency=20):
        super().__init__(batch_size=1)
        
        # Parámetros del agente
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.max_turns = max_turns
        self.imitation_weight = imitation_weight
        self.checkpoint_frequency = checkpoint_frequency
        
        # Crear el experto
        self.expert = ControllerAgent()
        
        # Estado de entrenamiento
        self.training_step = 0
        self.last_turn = 0
        self.current_episode = 0
        self.current_episode_reward = 0
        self.episode_rewards = []
        self.action_space = []
        
        # Variables de estado
        self.last_state = None 
        self.last_action = None
        self.previous_score = 0
        self.previous_cities = {}
        self.previous_units = {}
        self.previous_science = 0
        self.action_history = []
        
        # Configuración del dispositivo y modelo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_models()
        
        # Memoria de experiencias
        self.memory = ReplayMemory(50000)
        
        # Crear directorio para modelos si no existe
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Cargar modelo si existe
        self._load_model()
        
    def initialize_models(self):
        """Inicializa los modelos de red neuronal"""
        self.model = DQNetwork().to(self.device)
        self.target_model = DQNetwork().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
    def act(self, obs, info=None):
        """Selecciona una acción basada en el estado actual"""
        try:
            if info is None or 'available_actions' not in info:
                return None
                
            # Detectar cambio de turno y actualizar estado
            current_turn = info.get('turn', 0)
            if current_turn > self.last_turn:
                self._handle_turn_change(current_turn, info)
            
            # Obtener y filtrar acciones disponibles
            available_actions = info['available_actions']
            if not available_actions:
                return None

            # Obtener acción del experto
            expert_action = self.expert.act(obs, info)
                
            # Priorizar tipos de acciones basadas en estrategia
            action = self._select_strategic_action(obs, info, available_actions, expert_action)
            if action:
                # Guardar experiencia si tenemos estado previo
                if self.last_state is not None:
                    reward = self.calculate_reward(info)
                    self.memory.push(
                        self.last_state,
                        self.last_action,
                        self._state_to_tensor(obs, info, action[0], action[1]),
                        reward,
                        False,
                        expert_action
                    )
                
                # Actualizar estado actual
                self.last_state = self._state_to_tensor(obs, info, action[0], action[1])
                self.last_action = action
                
                # Actualizar historial y entrenar
                self._update_history(action, info)
                self._train_if_ready()
                return action
                
            return None
            
        except Exception as e:
            fc_logger.error(f"Error en act: {e}")
            return None

    def process_transition(self, state, action, next_state, reward, done, info, next_info):
        """Procesa una transición y la almacena en la memoria de experiencias"""
        try:
            # Convertir estados a tensores
            state_tensor = self._state_to_tensor(state, info, action[0] if action else None, 
                                               action[1] if action else None)
            next_state_tensor = self._state_to_tensor(next_state, next_info, action[0] if action else None,
                                                    action[1] if action else None)
            
            # Guardar experiencia
            self.memory.push(
                state_tensor,
                action,
                next_state_tensor,
                reward,
                done,
                None  # No usamos expert_action por ahora
            )
            
            # Actualizar recompensa del episodio
            self.current_episode_reward += reward
            
            # Entrenar si hay suficientes experiencias
            if len(self.memory) >= self.batch_size:
                self._train()
            
            # Actualizar red objetivo periódicamente
            if self.training_step % 100 == 0:
                self._update_target_model()
            
            # Actualizar epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Incrementar paso de entrenamiento
            self.training_step += 1
            
        except Exception as e:
            fc_logger.error(f"Error en process_transition: {e}")

    def _select_strategic_action(self, obs, info, available_actions, expert_action):
        """Selecciona acciones priorizando expansión y desarrollo"""
        # Orden de prioridad para acciones estratégicas
        priorities = ['city', 'unit', 'tech', 'gov']
        
        for ctrl_type in priorities:
            if ctrl_type not in available_actions:
                continue
                
            valid_actor_id, valid_actions = self.get_next_valid_actor(obs, info, ctrl_type)
            if not valid_actions:
                continue
                
            # Generar estado para el modelo
            state = self._state_to_tensor(obs, info, ctrl_type, valid_actor_id)
            
            # Decidir entre exploración, explotación o imitación
            if expert_action and random.random() < self.imitation_weight:
                # Seguir al experto
                if expert_action[0] == ctrl_type and expert_action[1] == valid_actor_id:
                    return expert_action
            elif random.random() < self.epsilon:
                # Exploración guiada por estrategia
                action_name = self._strategic_random_action(ctrl_type, valid_actions)
            else:
                # Explotación basada en el modelo
                action_name = self._get_best_action(state, valid_actions)
            
            if action_name:
                return (ctrl_type, valid_actor_id, action_name)
                
        return None

    def _strategic_random_action(self, ctrl_type, valid_actions):
        """Selecciona acciones aleatorias pero con preferencia estratégica"""
        try:
            actions = list(valid_actions.keys())
            
            if ctrl_type == 'city':
                # Priorizar construcción de colonos y mejoras importantes
                priority_actions = [a for a in actions if any(x in a.lower() for x in 
                                 ['settler', 'build', 'granary', 'library'])]
                if priority_actions:
                    return random.choice(priority_actions)
                    
            elif ctrl_type == 'unit':
                # Priorizar exploración y construcción de ciudades
                priority_actions = [a for a in actions if any(x in a.lower() for x in 
                                 ['build_city', 'explore', 'goto', 'road'])]
                if priority_actions:
                    return random.choice(priority_actions)
                    
            elif ctrl_type == 'tech':
                # Priorizar tecnologías útiles para expansión
                priority_actions = [a for a in actions if any(x in a.lower() for x in 
                                 ['alphabet', 'pottery', 'masonry', 'bronze'])]
                if priority_actions:
                    return random.choice(priority_actions)
            
            # Si no hay acciones prioritarias, elegir cualquiera excepto chat
            valid_actions = [a for a in actions if 'chat' not in a.lower()]
            return random.choice(valid_actions) if valid_actions else None
            
        except Exception as e:
            fc_logger.error(f"Error en _strategic_random_action: {e}")
            return None

    def _get_best_action(self, state, valid_actions):
        """Obtiene la mejor acción según el modelo actual"""
        try:
            with torch.no_grad():
                q_values = self.model(state)
                
            # Filtrar acciones no deseadas
            valid_actions = {k: v for k, v in valid_actions.items() if 'chat' not in k.lower()}
            if not valid_actions:
                return None
                
            # Convertir acciones a índices
            action_indices = [self.action_space.index(a) if a in self.action_space else -1 
                            for a in valid_actions.keys()]
            
            # Obtener valores Q para acciones válidas
            valid_q_values = [(i, q_values[0][i].item()) for i in action_indices if i >= 0]
            
            if not valid_q_values:
                return random.choice(list(valid_actions.keys()))
                
            # Seleccionar la acción con mayor valor Q
            best_action_idx = max(valid_q_values, key=lambda x: x[1])[0]
            return self.action_space[best_action_idx]
            
        except Exception as e:
            fc_logger.error(f"Error en _get_best_action: {e}")
            return None

    def calculate_reward(self, info):
        """Calcula la recompensa enfocada en expansión y progreso, con penalizaciones por pérdidas"""
        try:
            reward = 0
            
            # Recompensas y castigos por cambios en ciudades
            current_cities = info.get('city_data', {})
            new_cities = set(current_cities.keys()) - set(self.previous_cities.keys())
            lost_cities = set(self.previous_cities.keys()) - set(current_cities.keys())
            reward += len(new_cities) * 100  # Recompensa por nuevas ciudades
            reward -= len(lost_cities) * 150  # Castigo por perder ciudades
            
            # Recompensas y castigos por cambios en población
            for city_id in set(current_cities.keys()) | set(self.previous_cities.keys()):
                prev_size = self.previous_cities.get(city_id, {}).get('size', 0)
                current_size = current_cities.get(city_id, {}).get('size', 0) if city_id in current_cities else 0
                
                if current_size > prev_size:
                    reward += (current_size - prev_size) * 20  # Recompensa por crecimiento
                elif current_size < prev_size:
                    reward -= (prev_size - current_size) * 30  # Castigo por pérdida de población
            
            # Recompensa por unidades productivas
            current_units = info.get('unit_data', {})
            new_units = set(current_units.keys()) - set(self.previous_units.keys())
            reward += len(new_units) * 15
            
            # Recompensa por progreso científico
            current_science = info.get('science_progress', 0)
            if current_science > self.previous_science:
                reward += (current_science - self.previous_science) * 10
            
            # Actualizar estados previos
            self.previous_cities = current_cities
            self.previous_units = current_units
            self.previous_science = current_science
            
            return reward
            
        except Exception as e:
            fc_logger.error(f"Error en calculate_reward: {e}")
            return 0

    def _state_to_tensor(self, obs, info, ctrl_type, actor_id):
        """Convierte el estado del juego en un tensor para el modelo"""
        try:
            features = []
            
            # Características generales
            turn = info.get('turn', 0)
            features.extend([
                turn / self.max_turns,
                len(info.get('city_data', {})) / 30,  # Normalizar número de ciudades
                len(info.get('unit_data', {})) / 100,  # Normalizar número de unidades
            ])
            
            # Características específicas según tipo de controlador
            if ctrl_type == 'city':
                city_data = info['city_data'].get(actor_id, {})
                features.extend([
                    city_data.get('size', 0) / 30,
                    city_data.get('food_surplus', 0) / 100,
                    city_data.get('shield_surplus', 0) / 100,
                    city_data.get('trade_surplus', 0) / 100,
                ])
            elif ctrl_type == 'unit':
                unit_data = info['unit_data'].get(actor_id, {})
                features.extend([
                    unit_data.get('hp', 0) / 100,
                    unit_data.get('moves_left', 0) / unit_data.get('move_rate', 1),
                    1.0 if unit_data.get('can_build_city', False) else 0.0,
                ])
                
            # Padding si es necesario
            if len(features) < self.model.input_layer.in_features:
                features.extend([0] * (self.model.input_layer.in_features - len(features)))
            
            return torch.tensor(features, dtype=torch.float32, device=self.device)
            
        except Exception as e:
            fc_logger.error(f"Error en _state_to_tensor: {e}")
            return torch.zeros(self.model.input_layer.in_features, device=self.device)
            
    def _handle_turn_change(self, current_turn, info):
        """Maneja el cambio de turno y actualiza estado"""
        self.last_turn = current_turn
        
        # Guardar modelo periódicamente
        if current_turn % self.checkpoint_frequency == 0:
            self._save_model()
            
        # Actualizar epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def _update_history(self, action, info):
        """Actualiza el historial de acciones y estado"""
        self.action_history.append(action)
        if len(self.action_history) > 10:
            self.action_history.pop(0)
            
    def _train_if_ready(self):
        """Entrena el modelo si hay suficientes experiencias"""
        if len(self.memory) >= self.batch_size and self.training_step % 4 == 0:
            self._train()
            
        if self.training_step % 100 == 0:
            self._update_target_model()
            
        self.training_step += 1

    def _train(self):
        """Entrena el modelo con experiencias almacenadas"""
        try:
            # Obtener batch de experiencias
            experiences = self.memory.sample(self.batch_size)
            
            # Separar componentes
            states = torch.stack([e[0] for e in experiences])
            actions = [e[1] for e in experiences]
            next_states = torch.stack([e[2] for e in experiences])
            rewards = torch.tensor([e[3] for e in experiences], dtype=torch.float32, device=self.device)
            dones = torch.tensor([e[4] for e in experiences], dtype=torch.float32, device=self.device)
            expert_actions = [e[5] for e in experiences]
            
            # Calcular valores Q actuales
            current_q_values = self.model(states)
            
            # Calcular valores Q objetivo usando DQN
            with torch.no_grad():
                next_q_values = self.target_model(next_states)
                max_next_q = next_q_values.max(1)[0]
                target_q = rewards + (1 - dones) * self.gamma * max_next_q
            
            # Calcular pérdida de DQN
            action_indices = torch.tensor([self.action_space.index(a[2]) 
                                        if isinstance(a, tuple) and len(a) == 3 and a[2] in self.action_space 
                                        else -1 for a in actions], device=self.device)
            valid_indices = action_indices >= 0
            
            # Solo actualizar acciones válidas
            if valid_indices.any():
                dqn_loss = F.smooth_l1_loss(
                    current_q_values[valid_indices, action_indices[valid_indices]], 
                    target_q[valid_indices]
                )
            else:
                dqn_loss = torch.tensor(0.0, device=self.device)
            
            # Calcular pérdida de imitación
            expert_indices = torch.tensor([self.action_space.index(a[2]) 
                                        if isinstance(a, tuple) and len(a) == 3 and a[2] in self.action_space 
                                        else -1 for a in expert_actions], device=self.device)
            valid_expert = expert_indices >= 0
            
            if valid_expert.any():
                imitation_loss = F.cross_entropy(
                    current_q_values[valid_expert], 
                    expert_indices[valid_expert]
                )
            else:
                imitation_loss = torch.tensor(0.0, device=self.device)
            
            # Combinar pérdidas
            total_loss = (1 - self.imitation_weight) * dqn_loss + self.imitation_weight * imitation_loss
            
            # Optimizar
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
        except Exception as e:
            fc_logger.error(f"Error en _train: {e}")
            
    def _update_target_model(self):
        """Actualiza el modelo objetivo con los pesos del modelo principal"""
        self.target_model.load_state_dict(self.model.state_dict())
        
    def _save_model(self, force=False):
        """Guarda el modelo y metadatos"""
        try:
            # Crear un nombre de archivo con timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_filename = f'model_checkpoint_{self.training_step}_{timestamp}.pth'
            model_path = os.path.join(self.model_dir, model_filename)
            
            # Guardar el checkpoint actual
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'action_space': self.action_space,
                'training_step': self.training_step,
                'epsilon': self.epsilon
            }, model_path)
            
            # Crear un enlace simbólico al último modelo
            latest_path = os.path.join(self.model_dir, 'model_latest.pth')
            if os.path.exists(latest_path):
                os.remove(latest_path)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'action_space': self.action_space,
                'training_step': self.training_step,
                'epsilon': self.epsilon
            }, latest_path)
            
            # Mantener solo los 200 checkpoints más recientes
            checkpoints = glob.glob(os.path.join(self.model_dir, 'model_checkpoint_*.pth'))
            if len(checkpoints) > 200:
                # Ordenar por fecha de modificación (más antiguo primero)
                checkpoints.sort(key=os.path.getmtime)
                # Eliminar los más antiguos hasta tener solo 200
                for checkpoint in checkpoints[:-200]:
                    try:
                        os.remove(checkpoint)
                    except Exception as e:
                        fc_logger.error(f"Error eliminando checkpoint antiguo {checkpoint}: {e}")
            
        except Exception as e:
            fc_logger.error(f"Error guardando modelo: {e}")
            if force:
                # Intentar guardar al menos el último modelo en caso de error
                try:
                    latest_path = os.path.join(self.model_dir, 'model_latest.pth')
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'action_space': self.action_space,
                        'training_step': self.training_step,
                        'epsilon': self.epsilon
                    }, latest_path)
                except Exception as e2:
                    fc_logger.error(f"Error en guardado de emergencia: {e2}")
            
    def _load_model(self):
        """Carga el modelo y metadatos si existen"""
        try:
            model_path = os.path.join(self.model_dir, 'model_latest.pth')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.action_space = checkpoint['action_space']
                self.training_step = checkpoint['training_step']
                self.epsilon = checkpoint['epsilon']
                return True
            return False
            
        except Exception as e:
            fc_logger.error(f"Error cargando modelo: {e}")
            return False
