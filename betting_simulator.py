import numpy as np
import pandas as pd
from utils import get_features_labels, construct_full_match_sequence


class BettingSimulator:
    def __init__(self, model, data,
                 minute_odds=False,
                 start_odds=False,
                 is_lstm=False,
                 start_balance=100,
                 sure_bet_threshold=0.1,
                 stake_percent=0.03,
                 kelly_criterion=False,
                 odds_min=1.6, odds_max=2.4,
                 draw_bets=False):
        self.model = model
        self.is_lstm = is_lstm
        self.minute_odds = minute_odds
        self.start_odds = start_odds

        self.scaler = self.get_scaler()

        self.data = data
        self._prepare_events_dict()

        self.start_balance = start_balance
        self.balance = start_balance
        self.lowest_balance = start_balance
        self.highest_balance = start_balance

        self.bets = []
        self.events_bet_made = {}

        self.sure_bet_threshold = sure_bet_threshold
        self.stake_percent = stake_percent
        self.kelly_criterion = kelly_criterion
        self.odds_min = odds_min
        self.odds_max = odds_max
        self.draw_bets = draw_bets

    def simulate(self):
        for _, event in self.data.groupby('event.id'):
            for _, row in event.sort_values(by='minute').iterrows():
                event_id = row["event.id"]

                # only one bet on the same event
                if event_id in self.events_bet_made:
                    continue

                minute = row["minute"]
                home_goals = row["goals.home"]
                away_goals = row["goals.away"]
                home_odds = row["odds.home"]
                draw_odds = row["odds.draw"]
                away_odds = row["odds.away"]

                # skip minute with wrong odds
                if home_odds == 0 or draw_odds == 0 or away_odds == 0:
                    continue

                # scale input array
                if self.is_lstm:
                    event = self.events[event_id]
                    X = construct_full_match_sequence(event, minute, minute_odds=self.minute_odds,
                                                      start_odds=self.start_odds)
                else:
                    X, _ = get_features_labels(row, minute_odds=self.minute_odds,
                                               start_odds=self.start_odds).to_numpy().astype(np.float32)
                    X = np.multiply(X, self.scaler).reshape(1, X.shape[1])

                prediction = self.model.predict(X)
                outcome = row["result"]

                # probability distribution
                home_win_prob, draw_prob, away_win_prob = prediction[0]

                if home_win_prob != 0 and draw_prob != 0 and away_win_prob != 0:
                    # convert probabilities to bookmakers odds
                    exp_home_odds, exp_draw_odds, exp_away_odds = 1.0 / home_win_prob, 1.0 / draw_prob, 1.0 / away_win_prob
                else:
                    continue

                # home win bet opportunity
                if self._is_good_odds(home_odds, exp_home_odds):
                    print(
                        "Home win bet: event: %d | minute %d | odds: 1:%f x:%f 2:%f | score: %d-%d === Outcome: %d" % (
                            event_id, minute, home_odds, draw_odds, away_odds, home_goals, away_goals, outcome))
                    stake = self.get_stake_amount(home_win_prob, home_odds)

                    bet = {
                        "event_id": event_id,
                        "balance": self.balance,
                        "stake": stake,
                        "odds": home_odds,
                        "sure_bet": home_odds - exp_home_odds,
                        "type": 1,
                        "result": outcome,
                    }

                    self.bets.append(bet)
                    self.events_bet_made[event_id] = True

                    if outcome == 1:
                        self.balance += stake * (home_odds - 1.0)
                        if self.balance > self.highest_balance:
                            self.highest_balance = self.balance
                    else:
                        self.balance -= stake
                        if self.balance < self.lowest_balance:
                            self.lowest_balance = self.balance

                # away win bet opportunity
                elif self._is_good_odds(away_odds, exp_away_odds):
                    print("Away win bet: event: %d | minute %d | odds: 1:%f x:%f 2:%f |score: %d-%d === Outcome: %d" % (
                        event_id, minute, home_odds, draw_odds, away_odds, home_goals, away_goals, outcome))
                    stake = self.get_stake_amount(away_win_prob, away_odds)

                    bet = {
                        "event_id": event_id,
                        "balance": self.balance,
                        "stake": stake,
                        "odds": away_odds,
                        "sure_bet": away_odds - exp_away_odds,
                        "type": 2,
                        "result": outcome,
                    }

                    self.bets.append(bet)
                    self.events_bet_made[event_id] = True

                    if outcome == 2:
                        self.balance += stake * (away_odds - 1.0)
                        if self.balance > self.highest_balance:
                            self.highest_balance = self.balance
                    else:
                        self.balance -= stake
                        if self.balance < self.lowest_balance:
                            self.lowest_balance = self.balance

                # draw bet opportunity
                elif self.draw_bets and self._is_good_odds(draw_odds, exp_draw_odds):
                    print("Draw bet: event: %d | minute %d | odds: 1:%f x:%f 2:%f | score: %d-%d === Outcome: %d" % (
                        event_id, minute, home_odds, draw_odds, away_odds, home_goals, away_goals, outcome))
                    stake = self.get_stake_amount(away_win_prob, away_odds)

                    bet = {
                        "event_id": event_id,
                        "balance": self.balance,
                        "stake": stake,
                        "odds": draw_odds,
                        "sure_bet": draw_odds - exp_draw_odds,
                        "type": 0,
                        "result": outcome,
                    }

                    self.bets.append(bet)
                    self.events_bet_made[event_id] = True

                    if outcome == 0:
                        self.balance += stake * (draw_odds - 1.0)
                        if self.balance > self.highest_balance:
                            self.highest_balance = self.balance
                    else:
                        self.balance -= stake
                        if self.balance < self.lowest_balance:
                            self.lowest_balance = self.balance

    def summary(self):
        print("Start balance: %d" % self.start_balance)
        print("Lowest balance (-%.2f%%): %.2f" % ((self.start_balance - self.lowest_balance) / self.start_balance * 100,
                                                  self.lowest_balance))
        print(
            "Highest balance (+%.2f%%): %.2f" % ((self.highest_balance - self.start_balance) / self.start_balance * 100,
                                                 self.highest_balance))
        print("Balance after %d bets: %.2f" % (len(self.bets), self.balance))

        wins = 0
        losses = 0
        total_odds_sum = 0
        for bet in self.bets:
            if bet["type"] == bet["result"]:
                wins += 1
            else:
                losses += 1

            total_odds_sum += bet["odds"]

        accuracy = wins / len(self.bets) * 100
        print("Number of successful bets: ", wins)
        print("Average odds: ", total_odds_sum / len(self.bets))
        print("Win rate: %.2f%%" % accuracy)

    def _prepare_events_dict(self):
        events = {}
        for _, event in self.data.groupby('event.id'):
            event_id = event['event.id'].iloc[0]
            events[event_id] = event

        self.events = events

    def _is_good_odds(self, available_odds, expected_odds):
        if (available_odds - expected_odds > self.sure_bet_threshold) and \
                (self.odds_min <= available_odds <= self.odds_max):
            return True
        else:
            return False

    def calculate_kelly(self, win_prob, odds):
        stake_percent = ((win_prob * odds - 1) / (odds - 1))
        return self.balance * stake_percent

    def get_stake_amount(self, win_prob, odds):
        if self.kelly_criterion:
            stake = self.calculate_kelly(win_prob, odds)
        else:
            stake = self.balance * self.stake_percent

        return stake

    def get_scaler(self):
        n_features = 18
        if self.minute_odds:
            n_features += 3
        if self.start_odds:
            n_features += 3

        if self.is_lstm:
            return np.loadtxt("models/scaler.csv").astype(np.float32)[2:n_features]
        else:
            return np.loadtxt("models/scaler.csv").astype(np.float32)[:n_features]
