# *CivRealm*: A Learning and Reasoning Odyssey in *Civilization* for Decision-Making Agents

<div align="center">

[[Arxiv]](https://arxiv.org/abs/2401.10568)
[[PDF]](https://arxiv.org/pdf/2401.10568.pdf)
[[Docs]](https://bigai-ai.github.io/civrealm/)
[[LLM Agents]](https://github.com/bigai-ai/civrealm-llm-baseline)
[[Tensor Agent]](https://github.com/bigai-ai/civrealm-tensor-baseline)

[![Documentation Status](https://readthedocs.org/projects/openreview-py/badge/?version=latest)](<https://bigai-ai.github.io/civrealm>)
[![PyPI](https://img.shields.io/pypi/v/civrealm)](https://pypi.org/project/civrealm/)
[![PyPI - Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/bigai-ai/civrealm/dev/pyproject.toml)](https://pypi.org/project/civrealm/)
[![PyPI Status](https://pepy.tech/badge/civrealm)](https://pepy.tech/project/civrealm)
[![GitHub license](https://img.shields.io/github/license/bigai-ai/civrealm)](https://github.com/bigai-ai/civrealm/blob/main/LICENSE)

</div>

CivRealm is an interactive environment for the open-source strategy game [Freeciv-web](https://github.com/freeciv/freeciv-web), based on [Freeciv](https://www.freeciv.org/), a Civilization-like game. Within CivRealm, we provide interfaces for two typical types of agents: tensor-based reinforcement learning agents (see [Tensor-agent Repo](https://github.com/bigai-ai/civrealm-tensor-baseline)) based on the [Gymnasium](https://gymnasium.farama.org/) API, and language-based agents (see [LLM-agent Repo](https://github.com/bigai-ai/civrealm-llm-baseline)) driven by language models.

We also provide a set of tools for training and evaluating agents, as well as a set of baselines for both types of agents. We hope that CivRealm can serve as a testbed for the development and evaluation of agents that can learn and reason in complex environments. Detailed usage of the CivRealm API can be found in the [Documentation](https://bigai-ai.github.io/civrealm).

![Punic War](docs/assets/punic_war_base.jpg)

# Contents

- [About](#about)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Testing the Installation](#testing-the-installation)
  - [Single player mode (against built-in AIs)](#single-player-mode-against-built-in-ais)
  - [Multiplayer mode](#multiplayer-mode)
- [DQN Reinforcement Learning Agent](#dqn-reinforcement-learning-agent)
  - [Overview](#overview)
  - [Agent Structure](#agent-structure)
  - [Training Process](#training-process)
  - [Usage](#usage)
  - [Key Features](#key-features)
- [Trouble Shooting](#trouble-shooting)
- [Our Paper](#check-out-our-paper)
- [CivRealm en Español](#civrealm-en-español)

## About

CivRealm is developed based on [freeciv-bot](https://github.com/chris1869/freeciv-bot), dependent on [freeciv-web](<https://github.com/freeciv/freeciv-web>) and [FCIV-NET](<https://github.com/fciv-net/fciv-net>).
In the future, CivRealm will be maintained by BIGAI.

## Prerequisites

CivRealm requires Python `≥ 3.8` and docker. We have tested on Ubuntu 22.04, Mac OS X, and Windows. 

To test CivRealm on <http://localhost>, please follow the docker installation instructions on <https://bigai-ai.github.io/civrealm/getting_started/requirements.html>.

After starting the Freeciv-web service, you can connect to the Freeciv-web server via the host machine <a href="http://localhost:8080/">localhost:8080</a> using a standard browser.

## Installation

You can install the stable version of CivRealm by:

```bash
pip install civrealm
```

To install the latest version from the source code or contribute to the project, please follow the instructions below:

```bash
git clone git@github.com:bigai-ai/civrealm.git && cd civrealm
pip install -e .
```


<!-- 
### Update the freeciv-web image

Start the freeciv-web docker:

```bash
cd freeciv-web
docker compose up -d
```

Activate the civrealm virtual environment, and update the freeciv-web image:

```bash
update_freeciv_web_docker
```

Restart the freeciv-web container so that the change takes effect

```bash
cd freeciv-web
docker compose down
docker compose up -d
```
-->

## Testing the Installation

Before testing the installation, please make sure that the freeciv-web service is running. You can check the status of the freeciv-web service by running:

```bash
docker ps
```

You should see a docker container named `freeciv-web` running.

### Single player mode (against built-in AIs)

To test the installation, run the following command after installation. This will start a single player game against the built-in AIs with the default settings.

```bash
test_civrealm
```

!!! success
    If the installation is successful, the output should be similar to the following:

    ```bash
    Reset with port: 6300
    Step: 0, Turn: 1, Reward: 0, Terminated: False, Truncated: False, action: ('unit', 104, 'move NorthEast')
    Step: 1, Turn: 1, Reward: 0, Terminated: False, Truncated: False, action: ('unit', 117, 'move North')
    Step: 2, Turn: 1, Reward: 0, Terminated: False, Truncated: False, action: ('unit', 118, 'move North')
    Step: 3, Turn: 1, Reward: 0, Terminated: False, Truncated: False, action: ('unit', 119, 'move SouthEast')
    Step: 4, Turn: 1, Reward: 0, Terminated: False, Truncated: False, action: ('unit', 120, 'move SouthEast')
    ```

### Multiplayer mode

To test with multiple players, run the following command in a terminal to start the game with player `myagent`:

```bash
test_civrealm --minp=2 --username=myagent --client_port=6001
```

Then start another terminal and join the game with player `myagent1`:

```bash
test_civrealm --username=myagent1 --client_port=6001
```

## DQN Reinforcement Learning Agent

### Overview

The DQN (Deep Q-Network) agent is a reinforcement learning agent that learns to play Freeciv by interacting with the environment and learning from the rewards it receives. The agent is based on the [DQN algorithm](https://www.nature.com/articles/nature14236) introduced by DeepMind, which combines deep learning with Q-learning to learn optimal policies directly from high-dimensional sensory inputs.

The DQN agent in CivRealm uses separate neural networks for different controller types (units, cities, technology, government, etc.) and learns to take actions that maximize the expected future rewards. It employs an epsilon-greedy policy for exploration, where it randomly explores with probability epsilon and exploits its learned knowledge otherwise.

### Agent Structure

The DQN agent consists of the following components:

1. **Neural Networks**: Separate DQN networks for each controller type (units, cities, etc.)
```python
class DQNetwork(nn.Module):
    """Neural network for the DQN agent"""
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
```

2. **Replay Memory**: A memory buffer that stores experiences for training
```python
class ReplayMemory:
    """Replay memory to store agent experiences"""
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
```

3. **Exploration Strategy**: Epsilon-greedy policy with decaying epsilon
```python
# Exploration: take random action
if random.random() < self.epsilon:
    action_name = random.choice(list(valid_actions.keys()))
    return action_name
```

4. **Training Process**: Q-learning with target networks for stability
```python
# Calculate target Q value using target network
with torch.no_grad():
    next_q_values = target_model(next_states)
    max_next_q = next_q_values.max(1)[0]
    target_q = rewards + (1 - dones) * self.gamma * max_next_q
```

### Training Process

The DQN agent learns through the following process:

1. **Experience Collection**: The agent interacts with the Freeciv environment, taking actions and storing experiences (state, action, reward, next_state, done) in its replay memory.

2. **Batch Learning**: Periodically, the agent samples a batch of experiences from its replay memory and updates its Q-values using the Bellman equation:
   Q(s, a) = r + γ * max_a'(Q(s', a'))

3. **Progressive Improvement**: As training progresses, the agent's exploration rate (epsilon) decreases, leading it to rely more on its learned values and less on random exploration.

4. **Continuous Learning**: The agent can save its learned models and reload them to continue learning in future sessions, allowing for cumulative improvement over multiple sessions.

5. **Target Network Updates**: To stabilize learning, the agent maintains separate target networks that are periodically updated from the main networks.

### Usage

To use the DQN agent, run the following command:

```bash
test_civrealm_DQNA
```

This will start a game with the DQN agent. The agent will automatically load any previously trained models from the `dqn_models` directory and continue learning from where it left off. Training will continue indefinitely across multiple game episodes until manually stopped.

To stop the training gracefully (saving all models), press Ctrl+C. The agent will save its current state and exit.

### Key Features

1. **Continuous Learning**: The agent maintains learning across multiple episodes and sessions
```python
# Load previously trained models if they exist
for ctrl_type in controller_types:
    if agent._load_model(ctrl_type):
        models_loaded = True
```

2. **Safe Model Saving**: Models are saved periodically and on exit
```python
# Save models at the end of each episode
for ctrl_type in agent.models:
    agent._save_model(ctrl_type)
```

3. **Adaptive Model Architecture**: Neural networks automatically adjust to new action spaces
```python
if len(self.action_spaces[action_key]) > self.models[ctrl_type].network[-1].out_features:
    output_size = len(self.action_spaces[action_key])
    # Create new models with updated output size
```

4. **Incremental Learning**: The agent optimizes its policy through incremental updates
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

5. **Exploration Balance**: Automatic balance between exploration and exploitation
```python
# Adjust exploration rate
self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

## Trouble Shooting

The following are some common issues that you may encounter when running the code. If you encounter any other issues, please feel free to open an issue.

- If firefox keeps loading the page, please try to add the following line to `/etc/hosts`:

    ```bash
    127.0.0.1 maxcdn.bootstrapcdn.com
    127.0.0.1 cdn.webglstats.com
    ```

- If you see the following error when running `test_civrealm`,  please see [this solution](https://stackoverflow.com/questions/72405117/selenium-geckodriver-profile-missing-your-firefox-profile-cannot-be-loaded). If this does not solve the problem, please check `geckodriver.log` for more information.

    ```bash
    selenium.common.exceptions.WebDriverException: Message: Process unexpectedly closed with status 1
    ```

    One potential solution on Ubuntu 22.04 is:

    ```bash
    sudo apt install firefox-geckodriver
    ln -s /snap/bin/firefox.geckodriver geckodriver
    ```

- If you see the following error when setting `take_screenshot: True`, it is caused by snap version of Firefox. Please try [System Firefox installation](https://support.mozilla.org/en-US/kb/install-firefox-linux#w_install-firefox-from-mozilla-builds-for-advanced-users).

  ```bash
  Your Firefox profile cannot be loaded. 
  It may be missing or inaccessible.
  ```

- If the screenshot is not centered on the location of your first unit, it is because you are using multiple displays. Please ensure the Firefox browser for screenshot pops up on your primary display.

## Check out our paper

Our paper is available on [Arxiv](https://arxiv.org/abs/2401.10568). If you find our code or databases useful, please consider citing us:

```bibtex
@inproceedings{qi2024civrealm,
  title     = {CivRealm: A Learning and Reasoning Odyssey in Civilization for Decision-Making Agents},
  author    = {Siyuan Qi and Shuo Chen and Yexin Li and Xiangyu Kong and Junqi Wang and Bangcheng Yang and Pring Wong and Yifan Zhong and Xiaoyuan Zhang and Zhaowei Zhang and Nian Liu and Wei Wang and Yaodong Yang and Song-Chun Zhu},
  booktitle = {International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=UBVNwD3hPN}
}
```

# CivRealm en Español

# *CivRealm*: Una Odisea de Aprendizaje y Razonamiento en *Civilization* para Agentes de Toma de Decisiones

<div align="center">

[[Arxiv]](https://arxiv.org/abs/2401.10568)
[[PDF]](https://arxiv.org/pdf/2401.10568.pdf)
[[Docs]](https://bigai-ai.github.io/civrealm/)
[[LLM Agents]](https://github.com/bigai-ai/civrealm-llm-baseline)
[[Tensor Agent]](https://github.com/bigai-ai/civrealm-tensor-baseline)

[![Documentation Status](https://readthedocs.org/projects/openreview-py/badge/?version=latest)](<https://bigai-ai.github.io/civrealm>)
[![PyPI](https://img.shields.io/pypi/v/civrealm)](https://pypi.org/project/civrealm/)
[![PyPI - Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/bigai-ai/civrealm/dev/pyproject.toml)](https://pypi.org/project/civrealm/)
[![PyPI Status](https://pepy.tech/badge/civrealm)](https://pepy.tech/project/civrealm)
[![GitHub license](https://img.shields.io/github/license/bigai-ai/civrealm)](https://github.com/bigai-ai/civrealm/blob/main/LICENSE)

</div>

CivRealm es un entorno interactivo para el juego de estrategia de código abierto [Freeciv-web](https://github.com/freeciv/freeciv-web), basado en [Freeciv](https://www.freeciv.org/), un juego tipo Civilization. Dentro de CivRealm, proporcionamos interfaces para dos tipos típicos de agentes: agentes de aprendizaje por refuerzo basados en tensores (ver [Repositorio de Agente-Tensor](https://github.com/bigai-ai/civrealm-tensor-baseline)) basados en la API [Gymnasium](https://gymnasium.farama.org/), y agentes basados en lenguaje (ver [Repositorio de Agente-LLM](https://github.com/bigai-ai/civrealm-llm-baseline)) impulsados por modelos de lenguaje.

También proporcionamos un conjunto de herramientas para entrenar y evaluar agentes, así como una serie de bases de referencia para ambos tipos de agentes. Esperamos que CivRealm pueda servir como un banco de pruebas para el desarrollo y evaluación de agentes que puedan aprender y razonar en entornos complejos. El uso detallado de la API de CivRealm se puede encontrar en la [Documentación](https://bigai-ai.github.io/civrealm).

![Guerra Púnica](docs/assets/punic_war_base.jpg)

## Contenidos

- [Acerca de](#acerca-de)
- [Prerrequisitos](#prerrequisitos)
- [Instalación](#instalación)
- [Prueba de la Instalación](#prueba-de-la-instalación)
  - [Modo un jugador (contra IAs integradas)](#modo-un-jugador-contra-ias-integradas)
  - [Modo multijugador](#modo-multijugador)
- [Agente de Aprendizaje por Refuerzo DQN](#agente-de-aprendizaje-por-refuerzo-dqn)
  - [Visión General](#visión-general)
  - [Estructura del Agente](#estructura-del-agente)
  - [Proceso de Entrenamiento](#proceso-de-entrenamiento)
  - [Uso](#uso)
  - [Características Principales](#características-principales)
- [Solución de Problemas](#solución-de-problemas)
- [Consulta Nuestro Paper](#consulta-nuestro-paper)

## Acerca de

CivRealm se desarrolló basándose en [freeciv-bot](https://github.com/chris1869/freeciv-bot), dependiente de [freeciv-web](<https://github.com/freeciv/freeciv-web>) y [FCIV-NET](<https://github.com/fciv-net/fciv-net>).
En el futuro, CivRealm será mantenido por BIGAI.

## Prerrequisitos

CivRealm requiere Python `≥ 3.8` y docker. Hemos probado en Ubuntu 22.04, Mac OS X y Windows.

Para probar CivRealm en <http://localhost>, siga las instrucciones de instalación de docker en <https://bigai-ai.github.io/civrealm/getting_started/requirements.html>.

Después de iniciar el servicio Freeciv-web, puede conectarse al servidor Freeciv-web a través de la máquina host <a href="http://localhost:8080/">localhost:8080</a> usando un navegador estándar.

## Instalación

Puede instalar la versión estable de CivRealm mediante:

```bash
pip install civrealm
```

Para instalar la última versión desde el código fuente o contribuir al proyecto, siga las instrucciones a continuación:

```bash
git clone git@github.com:bigai-ai/civrealm.git && cd civrealm
pip install -e .
```

## Prueba de la Instalación

Antes de probar la instalación, asegúrese de que el servicio freeciv-web esté funcionando. Puede verificar el estado del servicio freeciv-web ejecutando:

```bash
docker ps
```

Debería ver un contenedor docker llamado `freeciv-web` en ejecución.

### Modo un jugador (contra IAs integradas)

Para probar la instalación, ejecute el siguiente comando después de la instalación. Esto iniciará un juego de un solo jugador contra las IAs integradas con la configuración predeterminada.

```bash
test_civrealm
```

!!! success
    Si la instalación es exitosa, la salida debería ser similar a la siguiente:

    ```bash
    Reset with port: 6300
    Step: 0, Turn: 1, Reward: 0, Terminated: False, Truncated: False, action: ('unit', 104, 'move NorthEast')
    Step: 1, Turn: 1, Reward: 0, Terminated: False, Truncated: False, action: ('unit', 117, 'move North')
    Step: 2, Turn: 1, Reward: 0, Terminated: False, Truncated: False, action: ('unit', 118, 'move North')
    Step: 3, Turn: 1, Reward: 0, Terminated: False, Truncated: False, action: ('unit', 119, 'move SouthEast')
    Step: 4, Turn: 1, Reward: 0, Terminated: False, Truncated: False, action: ('unit', 120, 'move SouthEast')
    ```

### Modo multijugador

Para probar con múltiples jugadores, ejecute el siguiente comando en una terminal para iniciar el juego con el jugador `myagent`:

```bash
test_civrealm --minp=2 --username=myagent --client_port=6001
```

Luego inicie otra terminal y únase al juego con el jugador `myagent1`:

```bash
test_civrealm --username=myagent1 --client_port=6001
```

## Agente de Aprendizaje por Refuerzo DQN

### Visión General

El agente DQN (Deep Q-Network) es un agente de aprendizaje por refuerzo que aprende a jugar Freeciv mediante la interacción con el entorno y el aprendizaje de las recompensas que recibe. El agente está basado en el [algoritmo DQN](https://www.nature.com/articles/nature14236) introducido por DeepMind, que combina aprendizaje profundo con Q-learning para aprender políticas óptimas directamente a partir de entradas sensoriales de alta dimensión.

El agente DQN en CivRealm utiliza redes neuronales separadas para diferentes tipos de controladores (unidades, ciudades, tecnología, gobierno, etc.) y aprende a tomar acciones que maximizan las recompensas futuras esperadas. Emplea una política epsilon-greedy para la exploración, donde explora aleatoriamente con probabilidad epsilon y explota su conocimiento aprendido en caso contrario.

### Estructura del Agente

1. **Arquitectura de Red**: Redes neuronales separadas para diferentes controladores
2. **Buffer de Repetición**: Almacena experiencias (estado, acción, recompensa, siguiente_estado)
3. **Política ε-greedy**: Balance entre exploración y explotación:
```python
if random.random() < self.epsilon:
    action_name = random.choice(list(valid_actions.keys()))
    return action_name
```

4. **Proceso de Entrenamiento**: Q-learning con redes objetivo para estabilidad:
```python 
# Calcula el valor Q objetivo usando la red objetivo
with torch.no_grad():
    next_q_values = target_model(next_states)
    max_next_q = next_q_values.max(1)[0]
    target_q = rewards + (1 - dones) * self.gamma * max_next_q
```

### Proceso de Entrenamiento

El agente DQN aprende a través del siguiente proceso:

1. **Recolección de Experiencias**: El agente interactúa con el entorno Freeciv, tomando acciones y almacenando experiencias (estado, acción, recompensa, siguiente_estado, terminado) en su memoria de repetición.

2. **Aprendizaje por Lotes**: Periódicamente, el agente muestrea un lote de experiencias de su memoria de repetición y actualiza sus valores Q usando la ecuación de Bellman:
   Q(s, a) = r + γ * max_a'(Q(s', a'))

3. **Mejora Progresiva**: A medida que avanza el entrenamiento, la tasa de exploración del agente (epsilon) disminuye, llevándolo a confiar más en sus valores aprendidos y menos en la exploración aleatoria.

4. **Aprendizaje Continuo**: El agente puede guardar sus modelos aprendidos y recargarlos para continuar aprendiendo en sesiones futuras, permitiendo una mejora acumulativa a lo largo de múltiples sesiones.

5. **Actualizaciones de Red Objetivo**: Para estabilizar el aprendizaje, el agente mantiene redes objetivo separadas que se actualizan periódicamente desde las redes principales.

### Uso

Para usar el agente DQN, ejecute el siguiente comando:

```bash
test_civrealm_DQNA
```

Esto iniciará un juego con el agente DQN. El agente cargará automáticamente cualquier modelo previamente entrenado desde el directorio `dqn_models` y continuará aprendiendo desde donde lo dejó. El entrenamiento continuará indefinidamente a través de múltiples episodios del juego hasta que se detenga manualmente.

Para detener el entrenamiento de forma segura (guardando todos los modelos), presione Ctrl+C. El agente guardará su estado actual y saldrá.

### Características Principales

1. **Aprendizaje Multi-Dominio**: 
   - Maneja diferentes aspectos del juego (unidades, ciudades, diplomacia)
   - Redes especializadas para cada dominio

2. **Exploración Adaptativa**:
   - Política epsilon-greedy con decaimiento
   - Balance automático entre exploración y explotación
```python
self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

3. **Persistencia del Modelo**:
   - Guardado y carga automática de modelos
   - Entrenamiento continuo entre sesiones

4. **Estabilidad de Entrenamiento**:
   - Uso de redes objetivo
   - Buffer de repetición para muestreo de experiencias
   - Actualizaciones graduales de parámetros

5. **Monitoreo de Rendimiento**:
   - Seguimiento de recompensas acumuladas
   - Métricas de progreso del juego
   - Visualización de estados del agente

## Solución de Problemas

Los siguientes son algunos problemas comunes que puede encontrar al ejecutar el código. Si encuentra otros problemas, no dude en abrir un issue.

- Si firefox sigue cargando la página, intente agregar la siguiente línea a `/etc/hosts`:

    ```bash
    127.0.0.1 maxcdn.bootstrapcdn.com
    127.0.0.1 cdn.webglstats.com
    ```

- Si ve el siguiente error al ejecutar `test_civrealm`, consulte [esta solución](https://stackoverflow.com/questions/72405117/selenium-geckodriver-profile-missing-your-firefox-profile-cannot-be-loaded). Si esto no resuelve el problema, verifique `geckodriver.log` para obtener más información.

    ```bash
    selenium.common.exceptions.WebDriverException: Message: Process unexpectedly closed with status 1
    ```

    Una posible solución en Ubuntu 22.04 es:

    ```bash
    sudo apt install firefox-geckodriver
    ln -s /snap/bin/firefox.geckodriver geckodriver
    ```

- Si ve el siguiente error al configurar `take_screenshot: True`, es causado por la versión snap de Firefox. Intente [Instalación de Firefox del Sistema](https://support.mozilla.org/en-US/kb/install-firefox-linux#w_install-firefox-from-mozilla-builds-for-advanced-users).

  ```bash
  Your Firefox profile cannot be loaded. 
  It may be missing or inaccessible.
  ```

- Si la captura de pantalla no está centrada en la ubicación de su primera unidad, es porque está utilizando múltiples pantallas. Asegúrese de que el navegador Firefox para captura de pantalla aparezca en su pantalla principal.

## Consulta Nuestro Paper

Nuestro paper está disponible en [Arxiv](https://arxiv.org/abs/2401.10568). Si encuentra útil nuestro código o bases de datos, considere citarnos:

```bibtex
@inproceedings{qi2024civrealm,
  title     = {CivRealm: A Learning and Reasoning Odyssey in Civilization for Decision-Making Agents},
  author    = {Siyuan Qi and Shuo Chen y Yexin Li y Xiangyu Kong y Junqi Wang y Bangcheng Yang y Pring Wong y Yifan Zhong y Xiaoyuan Zhang y Zhaowei Zhang y Nian Liu y Wei Wang y Yaodong Yang y Song-Chun Zhu},
  booktitle = {International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=UBVNwD3hPN}
}
```
