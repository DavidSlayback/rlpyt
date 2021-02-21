from rlpyt.utils.collections import namedarraytuple, AttrDict
import numpy as np
class TrajInfoVec(AttrDict):
    """
    Because it inits as an AttrDict, this has the methods of a dictionary,
    e.g. the attributes can be iterated through by traj_info.items()
    Intent: all attributes not starting with underscore "_" will be logged.
    (Can subclass for more fields.)
    Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase.

    This is a version for use with vectorized environments, all inputs are batched
    I'm also not taking traj_done into account, since it doesn't apply to the environments I'll use this for
    """

    _discount = 1  # Leading underscore, but also class attr not in self.__dict__.

    def __init__(self, B=1, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self.Length = np.zeros(B)
        self.Return = np.zeros(B)
        self.NonzeroRewards = np.zeros(B)
        self.DiscountedReturn = np.zeros(B)
        self._cur_discount = np.ones(B)

    def step(self, observation, action, reward, done, agent_info, env_info, reset_dones=False):
        """ Step AND take dones into account"""
        self.Length += 1
        self.Return+= reward
        self.NonzeroRewards[reward != 0] += 1
        self.DiscountedReturn += self._cur_discount * reward
        self._cur_discount *= self._discount
        if reset_dones:
            self.reset_dones(done)

    def reset_dones(self, done):
        l_A, r_A, nr_A, dr_A, _cd_A = self.Length[done], self.Return[done], self.NonzeroRewards[done], self.DiscountedReturn[done], self._cur_discount[done]
        completed_infos = [AttrDict(Length=l,
                                    Return=r,
                                    NonzeroRewards=nr,
                                    DiscountedReturn=dr,
                                    _cur_discount=_cd,
                                    _discount=self._discount) for l, r, nr, dr, _cd in zip(l_A, r_A, nr_A, dr_A, _cd_A)]
        self.Length[done], self.Return[done], self.NonzeroRewards[done], self.DiscountedReturn[done], self._cur_discount[done] = 0, 0., 0., 0., 1.
        return completed_infos

    def terminate(self, done):
        return self.reset_dones(done)

class TrajInfoVecGPU(AttrDict):
    """
    Because it inits as an AttrDict, this has the methods of a dictionary,
    e.g. the attributes can be iterated through by traj_info.items()
    Intent: all attributes not starting with underscore "_" will be logged.
    (Can subclass for more fields.)
    Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase.

    This is a version for use with vectorized environments, all inputs are batched cuda tensors
    I'm also not taking traj_done into account, since it doesn't apply to the environments I'll use this for
    """

    _discount = 1  # Leading underscore, but also class attr not in self.__dict__.

    def __init__(self, B=1, **kwargs):
        import torch
        super().__init__(**kwargs)  # (for AttrDict behavior)
        device = 'cuda'
        self.Length = torch.zeros(B, device=device)
        self.Return = torch.zeros_like(self.Length)
        self.NonzeroRewards = torch.zeros_like(self.Length)
        self.DiscountedReturn = torch.zeros_like(self.Length)
        self._cur_discount = torch.ones_like(self.Length)

    def step(self, observation, action, reward, done, agent_info, env_info, reset_dones=False):
        """ Step AND take dones into account if reset_dones is True"""
        self.Length += 1
        self.Return+= reward
        self.NonzeroRewards[reward != 0] += 1
        self.DiscountedReturn += self._cur_discount * reward
        self._cur_discount *= self._discount
        if reset_dones:
            self.reset_dones(done)

    def reset_dones(self, done):
        done = done.bool()
        l_A, r_A, nr_A, dr_A, _cd_A = self.Length[done].cpu().numpy(), self.Return[done].cpu().numpy(), \
                            self.NonzeroRewards[done].cpu().numpy(), self.DiscountedReturn[done].cpu().numpy(), \
                            self._cur_discount[done].cpu().numpy()
        completed_infos = [AttrDict(Length=l,
                                    Return=r,
                                    NonzeroRewards=nr,
                                    DiscountedReturn=dr,
                                    _cur_discount=_cd,
                                    _discount=self._discount) for l, r, nr, dr, _cd in zip(l_A, r_A, nr_A, dr_A, _cd_A)]
        self.Length[done], self.Return[done], self.NonzeroRewards[done], self.DiscountedReturn[done], self._cur_discount[done] = 0, 0., 0., 0., 1.
        return completed_infos

    def terminate(self, dones):
        return self.reset_dones(dones)