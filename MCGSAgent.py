import stratega
import numpy as np
import math
from copy import deepcopy
from MCGSGraph import Graph
from utils import Timer
from heuristics import *

'''
Current problems:
    1. The invalid actions are caused when there are no selectable nodes in the frontier
    when this happens, we return the root node and the root node action might not be valid at that state

    We need to find a better way to select nodes rather than maintaining a frontier

'''

class MCGSAgent(stratega.Agent):

    def __init__(self, seed, budget_type="MAX_FM_CALLS"):
        stratega.Agent.__init__(self, "MCGSAgent")
        self.seed = seed 
        self.budget_type = budget_type

    def init(self, gs, forward_model, timer):
        print("init MCGSAgent")
        self.random = np.random.RandomState(self.seed)
        self.graph = Graph(self.seed)
        self.node_counter = 0
        self.edge_counter = 0
        self.num_rollouts = 8

        self.use_opponent_model = True 
        self.root_node = Node(id=self.get_observation(gs), parent=None, is_leaf=True, action=None, reward=0, visits=0)
        self.add_node(self.root_node)
        self.root_node.chosen = True 
 
        self.num_simulations = 0
        self.forward_model_calls = 0
        self.max_forward_model_calls = 1000

        self.num_iterations = 0
        self.max_iterations = 100

        self.timer = Timer()
        self.max_time_ms = 1000

        self.heuristic = GeneralHeuristic(gs)


    def is_budget_over(self):
        if self.budget_type == "MAX_FM_CALLS":
            return self.forward_model_calls >= self.max_forward_model_calls
        elif self.budget_type == "MAX_ITERATIONS":
            return self.num_iterations >= self.max_iterations
        elif self.budget_type == "MAX_TIME_MS":
            return self.timer.elapsed_milliseconds() >= self.max_time_ms
        else:
            return False 

    def reset_budget(self):
        self.forward_model_calls = 0
        self.num_iterations = 0
        self.timer = Timer()

    def evaluate_state(self, forward_model, gs, player_id):
        return self.heuristic.evaluate_gamestate(forward_model, gs, player_id)    

    def is_game_over(self, gs):
        return gs.is_game_over()

    def get_observation(self, gs):
        return gs.print_board()

    def get_opponent_id(self):
        if self.get_player_id() == 0:
            return 1 
        else:
            return 0

    def compute_action(self, gs, forward_model, timer, draw_graph=False):
        self.reset_budget()
        possible_actions = forward_model.generate_actions(gs, self.get_player_id())
       
        if len(possible_actions) == 1:
            return stratega.ActionAssignment.from_single_action(possible_actions[0])

        action = self.plan(gs, forward_model)
            
        if action.validate(gs) == False:
                print("invalid action... ending turn")
                action = possible_actions[-1]

        action_assignment = stratega.ActionAssignment.from_single_action(action)
                
        if draw_graph:
            self.graph.draw_graph()

        return action_assignment


    def plan(self, gs, forward_model): 
        self.set_root_node(gs)
        self.graph.reroute_all()

        while not self.is_budget_over():

            selection_env = deepcopy(gs)
            node = self.selection(selection_env, forward_model)
         
            children, actions_to_children = self.expansion(node, selection_env, forward_model)
                 
            for idx in range(len(children)):

                child_average_reward = self.simulation(actions_to_children[idx], selection_env, forward_model)
                self.num_simulations += 1      
                self.back_propagation(children[idx], child_average_reward)
                
            self.num_iterations += 1

        best_node, action = self.select_best_node(self.root_node)
       
        return action

    def selection(self, env, forward_model):

        if self.root_node.is_leaf:
            return self.root_node

        node = self.graph.select_frontier_node()

        # found the problem
        if node is None:
            return self.root_node

        selected_node = self.go_to_node(node, env, forward_model)
        return selected_node

    def go_to_node(self, destination_node, env, forward_model): 


        observation = self.get_observation(env)
        node = self.graph.get_node_info(observation)

        reached_destination = False

        while self.graph.has_path(node, destination_node) and not reached_destination:
            
            observations, actions = self.graph.get_path(node, destination_node)

            for idx, action in enumerate(actions):

                previous_observation = self.get_observation(env)
                parent_node = self.graph.get_node_info(previous_observation)
                
                forward_model.advance_gamestate(env, action)
                self.forward_model_calls += 1
                reward = self.evaluate_state(forward_model, env, self.get_player_id())
               
                current_observation = self.get_observation(env)
                
                if not self.graph.has_node(current_observation):
                    self.add_new_observation(current_observation, parent_node, action, reward)

                if not self.graph.has_edge_by_nodes(parent_node, self.graph.get_node_info(current_observation)):
                    self.add_edge(parent_node, self.graph.get_node_info(current_observation), action, reward)

                if observations[idx + 1] != current_observation:
                    node = self.graph.get_node_info(current_observation)
                    break
              
                if self.get_observation(env) == destination_node.id:
                    reached_destination = True
                    break


        return self.graph.get_node_info(self.get_observation(env))
      

    def expansion(self, node, env, forward_model):

        new_nodes = []
        actions_to_new_nodes = []

        if node.is_leaf:
            node.is_leaf = False
            self.graph.remove_from_frontier(node)

        actions = forward_model.generate_actions(env, self.get_player_id())

        for action in actions:

            expansion_env = deepcopy(env)

            forward_model.advance_gamestate(expansion_env, action)
            self.forward_model_calls += 1
            reward = self.evaluate_state(forward_model, expansion_env, self.get_player_id())
          
            current_observation = self.get_observation(expansion_env)
            child, reward = self.add_new_observation(current_observation, node, action, reward)

            if child is not None:
                new_nodes.append(child)
                actions_to_new_nodes.append(action)

        return new_nodes, actions_to_new_nodes


    def simulation(self, action_to_node, env, forward_model):
        rewards = []
        simulation_env = deepcopy(env)
        forward_model.advance_gamestate(simulation_env, action_to_node)
        self.forward_model_calls += 1    

        for i in range(self.num_rollouts):
            average_reward = self.rollout(simulation_env, forward_model)
            rewards.append(average_reward)

        return np.mean(rewards)

   
    def rollout(self, env, forward_model):
        cum_reward = 0
        rollout_env = deepcopy(env)

        while True:
            actions = forward_model.generate_actions(rollout_env, self.get_player_id())

            if len(actions) == 0:
                break

            if self.is_budget_over():
                break 

            if self.is_game_over(rollout_env):
                break 

            random_action = self.random.choice(actions)
            forward_model.advance_gamestate(rollout_env, random_action)
            self.forward_model_calls += 1
            reward = self.evaluate_state(forward_model, rollout_env, self.get_player_id())
            cum_reward += reward 

            if self.use_opponent_model:
                rollout_env.set_current_tbs_player(self.get_opponent_id())

                opponent_actions = forward_model.generate_actions(rollout_env, self.get_opponent_id())
                random_opponent_action = self.random.choice(opponent_actions)
                forward_model.advance_gamestate(rollout_env, random_opponent_action)

                rollout_env.set_current_tbs_player(self.get_player_id())
       
        return cum_reward

    def back_propagation(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_value += reward 
            node = node.parent

    def set_root_node(self, gs):
        old_root_node = self.root_node
        new_root_id = self.get_observation(gs)

        self.root_node = self.graph.get_node_info(new_root_id)
        self.graph.set_root_node(self.root_node)

        if self.root_node.id != old_root_node.id:
            self.root_node.chosen = True
            self.root_node.parent = None

            if self.graph.has_path(self.root_node, old_root_node):
                self.graph.reroute_path(self.root_node, old_root_node)
                old_root_node.action = self.graph.get_edge_info(old_root_node.parent, old_root_node).action

    def add_new_observation(self, current_observation, parent_node, action, reward):

        new_node = None

        if current_observation != parent_node.id:  
            if self.graph.has_node(current_observation) is False:  
                child = Node(id=current_observation, parent=parent_node,
                             is_leaf=True, action=action, reward=reward, visits=0)
                self.add_node(child)
                new_node = child
            else:
                child = self.graph.get_node_info(current_observation)

                # if child.is_leaf: # enable for FMC optimisation, comment for full exploration
                #     new_node = child

            self.add_edge(parent_node, child, action, reward)
   
        return new_node, reward
    

    def select_best_node(self, node):

        best_node = self.graph.get_best_node()

        if best_node is None:
            return self.root_node, self.root_node.action

        while best_node.parent != self.root_node:
            best_node = best_node.parent

        edge = self.graph.get_edge_info(node, best_node)  

        return best_node, edge.action


    def add_node(self, node):
        if not self.graph.has_node(node):
            self.graph.add_node(node)
            self.graph.add_to_frontier(node)
            self.node_counter += 1


    def add_edge(self, parent_node, child_node, action, reward):

        edge = Edge(id=self.edge_counter, node_from=parent_node, node_to=child_node,
                    action=action, reward=reward)

        if not self.graph.has_edge(edge):
            self.graph.add_edge(edge)
            self.edge_counter += 1

           
        if child_node.unreachable is True and child_node != self.root_node:  
            child_node.set_parent(parent_node)
            child_node.action = action
            child_node.unreachable = False

        return edge

class Node:

    def __init__(self, id, parent, is_leaf, action, reward, visits):

        self.id = id
        self.parent = None
        self.set_parent(parent)

        self.action = action
        self.total_value = reward
        self.visits = visits
        self.is_leaf = is_leaf

        self.chosen = False
        self.unreachable = False

    def uct_value(self):
        c = 1 / math.sqrt(2)
        ucb = c * math.sqrt(math.log(self.parent.visits + 1) / self.visits)
        return self.value() + ucb

    def value(self):
        if self.visits == 0:
            return 0
        else:
            return self.total_value / self.visits

    def trajectory_from_root(self):

        action_trajectory = []
        current_node = self

        while current_node.parent is not None:
            action_trajectory.insert(0, current_node.action)
            current_node = current_node.parent

        return action_trajectory

    def reroute(self, path, actions):
        parent_order = list(reversed(path))
        actions_order = list(reversed(actions))
        node = self

        for i in range(len(parent_order) - 1):
            if node.parent != parent_order[i + 1]:
                node.set_parent(parent_order[i + 1])
            node.action = actions_order[i]
            node = parent_order[i + 1]

    def set_parent(self, parent):
        self.parent = parent

    def __hash__(self):
        return hash(self.id)


class Edge:

    def __init__(self, id, node_from, node_to, action, reward):
        self.id = id
        self.node_from = node_from
        self.node_to = node_to
        self.action = action
        self.reward = reward

    def __hash__(self):
        return hash(self.id)


