import numpy as np

from src.do_not_touch.result_structures import (
    ValueFunction,
    PolicyAndValueFunction,
)


class DynamicProgramming:
    def __init__(self):
        self.theta = 0.0000001
        self.gamma = 0.99999
    
    def policy_evaluation(
        self,
        states,
        actions,
        rewards,
        transition_probability_function,
        policy_function,
    ) -> ValueFunction:
        V = {state: 0.0 for state in states}
        
        while True:
            delta = 0.0
            
            for state in states:
                old_v = V[state]
                total = 0.0
                
                for action in actions:
                    total_inter = 0.0
                    
                    for state_policy in states:
                        for reward in range(len(rewards)):
                            total_inter += (
                                transition_probability_function(
                                    state,
                                    action,
                                    state_policy,
                                    reward,
                                ) * (
                                    rewards[reward]
                                    + self.gamma
                                    * V[state_policy]
                                )
                            )
                    
                    total_inter = policy_function(state, action) * total_inter
                    total += total_inter
                
                V[state] = total
                delta = max(delta, np.abs(V[state] - old_v))
            
            if delta < self.theta:
                return V
    
    def policy_iteration(
        self,
        states,
        actions,
        rewards,
        transition_probability_function,
    ) -> PolicyAndValueFunction:
        policy_function = {
            state: {
                action: 1 / len(actions)
                for action
                in actions
            }
            for state
            in states
        }
        
        V = self.policy_evaluation(
            states,
            actions,
            rewards,
            transition_probability_function,
            lambda state, action: policy_function[state][action],
        )
        
        while True:
            policy_stable = True
            
            for state in states:
                greatest_pi_of_s_key = max(
                    policy_function[state],
                    key=policy_function[state].get,
                )
                
                best_action = actions[np.argmax([
                    sum(
                        [
                            (
                                transition_probability_function(
                                    state,
                                    action,
                                    state_policy,
                                    reward,
                                ) * (
                                    rewards[reward]
                                    + self.gamma
                                    * V[state_policy]
                                )
                            )
                            for state_policy
                            in states
                            for reward
                            in range(len(rewards))
                        ]
                    )
                    for action
                    in actions
                ])]
                
                policy_function[state] = {
                    action: 1
                    if action == best_action
                    else 0
                    for action
                    in actions
                }
                
                if greatest_pi_of_s_key != best_action:
                    policy_stable = False
            
            if policy_stable:
                break
            else:
                V = self.policy_evaluation(
                    states,
                    actions,
                    rewards,
                    transition_probability_function,
                    lambda state, action: policy_function[state][action],
                )
        
        return PolicyAndValueFunction(policy_function, V)
    
    def value_iteration(
        self,
        states,
        actions,
        rewards,
        transition_probability_function,
    ) -> PolicyAndValueFunction:
        V = {state: 0 for state in states}
        
        while True:
            delta = 0
            
            for state in states:
                old_v_state = V[state]
                V[state] = max(
                    [
                        sum(
                            [
                                (
                                    transition_probability_function(
                                        state,
                                        action,
                                        state_policy,
                                        reward,
                                    ) * (
                                        rewards[reward]
                                        + self.gamma
                                        * V[state_policy]
                                    )
                                )
                                for state_policy
                                in states
                                for reward
                                in range(len(rewards))
                            ]
                        )
                        for action
                        in actions
                    ]
                )
                
                delta = max(delta, np.abs(old_v_state - V[state]))
            
            if delta < self.theta: break
        
        policy_function = {
            state: {
                action: 0
                for action
                in actions
            }
            for state
            in states
        }
        
        for state in states:
            policy_function[state][np.argmax([
                sum(
                    [
                        (
                            transition_probability_function(
                                state,
                                action,
                                state_policy,
                                reward
                            ) * (
                                rewards[reward]
                                + self.gamma
                                * V[state_policy]
                            )
                        )
                        for state_policy
                        in states
                        for reward
                        in range(len(rewards))
                    ]
                )
                for action
                in actions
            ])] = 1
        
        return PolicyAndValueFunction(policy_function, V)
