from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import wrappers
from gymnasium.spaces import Discrete, Dict as DictSpace, Box

from .constants import ACTION_SPACE_SIZE, Action, Phase, Role
from .game_state import GameConfig, GameRNG, GameState, PlayerState
from .observation import build_observation, build_legal_action_mask, card_name


AGENTS_DEFAULT = [
    "player_0",
    "player_1",
    "player_2",
    "player_3",
]


def env(seed: int = 0, num_players: int = 4):
    e = SgsAecEnv(GameConfig(num_players=num_players, seed=seed))
    return wrappers.OrderEnforcingWrapper(e)


class SgsAecEnv(AECEnv):
    metadata = {"name": "sgs_aec_v0", "is_parallelizable": False}

    def __init__(self, config: GameConfig):
        super().__init__()
        self.config = config
        self.agents: List[str] = AGENTS_DEFAULT[: config.num_players]
        self.possible_agents = list(self.agents)
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.agents)}

        # observation and action spaces are stable per agent
        self._action_space = Discrete(ACTION_SPACE_SIZE)
        # Observation is a nested dict, we place a placeholder Box for compatibility
        self._observation_space = Box(low=0, high=255, shape=(1,), dtype=np.uint8)

        self.reset(seed=config.seed)

    # PettingZoo required API
    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is None:
            seed = self.config.seed
        self.np_random = np.random.default_rng(seed)
        rng = GameRNG(seed=seed, rng=self.np_random)
        self.state = self._init_game_state(rng)

        self.agents = self.possible_agents[:]
        self._agent_selector_reset()
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._update_all_infos()

    def _agent_selector_reset(self):
        self.agent_selection = self.agents[0] if self.agents else None

    def observe(self, agent: str):
        obs_dict = build_observation(self.state, agent)
        # We return a dummy Box obs; the real obs is in info for now to keep MVP simple for tests
        return np.array([0], dtype=np.uint8)

    def last(self, observe: bool = True):
        agent = self.agent_selection
        obs = self.observe(agent) if observe else None
        rew = self.rewards[agent]
        term = self.terminations[agent]
        trunc = self.truncations[agent]
        info = self.infos[agent]
        return obs, rew, term, trunc, info

    def step(self, action: int):
        agent = self.agent_selection
        if agent is None:
            return
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        legal = build_legal_action_mask(self.state, agent)
        if action < 0 or action >= ACTION_SPACE_SIZE or legal[action] == 0:
            # illegal action treated as PASS/CONFIRM depending on phase
            action = self._default_action_for_phase()

        events: List[Dict] = []
        self._advance_with_action(agent, action, events)

        # Update rewards (dense=0 for MVP)
        self.rewards = {a: 0.0 for a in self.agents}

        # Rotate to next agent for AEC iteration
        self._select_next_agent()
        self._update_all_infos(extra_events=events)

    def _default_action_for_phase(self) -> int:
        phase = self.state.current_phase
        if phase in (Phase.DISCARD, Phase.PREPARE, Phase.JUDGEMENT, Phase.DRAW, Phase.END):
            return int(Action.CONFIRM)
        return int(Action.PASS)

    def _select_next_agent(self):
        # In AEC, we select current player only; others receive observations as their turn arrives
        self.agent_selection = self.state.agent_order[self.state.current_agent_idx]

    # Core game flow
    def _init_game_state(self, rng: GameRNG) -> GameState:
        state = GameState(config=self.config, rng=rng)
        # Build a small deck: 1..3 cards repeated
        suits = [0, 1, 2, 3]
        deck: List[Tuple[int, int]] = []
        for _ in range(30):
            cid = rng.rng.integers(1, 4)
            s = int(rng.rng.choice(suits))
            deck.append((int(cid), s))
        rng.rng.shuffle(deck)
        state.deck = deck

        # Assign roles in 4P: LORD, REBEL, REBEL, LOYALIST
        roles = [Role.LORD, Role.REBEL, Role.REBEL, Role.LOYALIST][: self.config.num_players]
        rng.rng.shuffle(roles)

        for i, agent in enumerate(self.agents):
            state.players[agent] = PlayerState(seat=i, role=roles[i])
        state.agent_order = list(self.agents)
        state.current_agent_idx = 0
        state.current_phase = Phase.PREPARE
        state.turn_count = 1
        state.used_sha_in_turn = {a: False for a in self.agents}

        # initial draw: 4 cards each
        for agent in state.agent_order:
            self._draw_cards(state, agent, 4)
        return state

    def _draw_cards(self, state: GameState, agent: str, n: int):
        for _ in range(n):
            if not state.deck:
                break
            card = state.deck.pop()  # (cid, suit)
            state.players[agent].hand.append(card)

    def _advance_with_action(self, agent: str, action: int, events: List[Dict]):
        state = self.state
        phase = state.current_phase
        me = state.players[agent]

        if phase == Phase.PREPARE:
            events.append({"type": "phase", "phase": Phase.PREPARE.value, "agent": agent})
            state.current_phase = Phase.JUDGEMENT
            return
        if phase == Phase.JUDGEMENT:
            events.append({"type": "phase", "phase": Phase.JUDGEMENT.value, "agent": agent})
            state.current_phase = Phase.DRAW
            return
        if phase == Phase.DRAW:
            events.append({"type": "phase", "phase": Phase.DRAW.value, "agent": agent})
            self._draw_cards(state, agent, 2)
            state.current_phase = Phase.PLAY
            return
        if phase == Phase.PLAY:
            if action == Action.PASS:
                events.append({"type": "pass", "agent": agent})
                state.current_phase = Phase.DISCARD
                return
            if action == Action.PLAY_TAO:
                if me.hp < me.max_hp:
                    me.hp = min(me.max_hp, me.hp + 1)
                    self._consume_first_named_card(me, name="tao")
                    events.append({"type": "play", "card": "tao", "agent": agent})
                else:
                    events.append({"type": "noop", "reason": "hp_full"})
                return
            if action == Action.PLAY_SHA:
                if not self.state.used_sha_in_turn.get(agent, False) and self._consume_first_named_card(me, name="sha"):
                    self.state.used_sha_in_turn[agent] = True
                    events.append({"type": "play", "card": "sha", "agent": agent})
                else:
                    events.append({"type": "noop", "reason": "sha_limit_or_no_card"})
                return
            # default: end play phase
            state.current_phase = Phase.DISCARD
            return
        if phase == Phase.DISCARD:
            over = len(me.hand) - me.hp
            if over > 0:
                if action == Action.DISCARD_CARD and me.hand:
                    # discard the last card as a deterministic choice for MVP
                    dropped = me.hand.pop()
                    state.discard_pile.append(dropped)  # (cid, suit)
                    events.append({"type": "discard", "agent": agent, "count": 1})
                else:
                    events.append({"type": "rule_enforced_wait_discard"})
                return
            # hand size <= hp, confirm to end
            if action == Action.CONFIRM:
                events.append({"type": "phase", "phase": Phase.END.value, "agent": agent})
                state.current_phase = Phase.END
                return
            return
        if phase == Phase.END:
            # advance to next player's turn
            state.current_phase = Phase.PREPARE
            state.used_sha_in_turn[agent] = False
            state.current_agent_idx = (state.current_agent_idx + 1) % len(state.agent_order)
            if state.current_agent_idx == 0:
                state.turn_count += 1
            events.append({"type": "turn_end", "agent": agent})
            return

    def _consume_first_named_card(self, me: PlayerState, name: str) -> bool:
        # Scan hand indices and remove the first card with matching name
        for i, (cid, suit) in enumerate(me.hand):
            if card_name(cid) == name:
                card = me.hand.pop(i)
                self.state.discard_pile.append(card)
                return True
        return False

    def _update_all_infos(self, extra_events: Optional[List[Dict]] = None):
        for agent in self.agents:
            obs_struct = build_observation(self.state, agent)
            mask = build_legal_action_mask(self.state, agent)
            self.infos[agent] = {
                "legal_action_mask": mask,
                "events": extra_events or [],
                "rng": self.state.to_info_rng_hash(),
                "observation_struct": obs_struct,  # full struct for API/UI
            }