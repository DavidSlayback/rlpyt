

def gen_pseudo_reward():
    """
    Generate the reward for entropy between action policies of each pair of options (want diverse option behavior
    List all combinations of 2 options
    In discrete:
        Get logits, take softmax, clip, and get either joint or cross-entropy
    Softmax used in continuous control to keep rewards positive

    """
    pass
def gen_pseudo_reward_old(pi, ob, num_options, stochastic=True, cross=True):
    if num_options==1:
        return 0
    else:
        cum_entropy, joint_entropy = 0,0
        combinations = list(itertools.combinations(range(num_options), 2))
        for i in range(len(combinations)):
            sampled_op1 = combinations[i][0]
            sampled_op2 = combinations[i][1]
            if isinstance(pi, cnn.CnnPolicy):
                x1, _, _, _= pi.act(stochastic, ob, sampled_op1)
                x2, _, _, _ = pi.act(stochastic, ob, sampled_op2)
                logits_op1 = pi.get_logits(stochastic, ob, sampled_op1)[0]
                logits_op2 = pi.get_logits(stochastic, ob, sampled_op2)[0]
                pd_op1 = softmax(logits_op1)
                pd_op2 = softmax(logits_op2)
                pd_op1 = np.clip(pd_op1,1e-20, 1.0)
                pd_op2 = np.clip(pd_op2,1e-20, 1.0)
                if cross:
                    cum_entropy += -np.sum(pd_op1*np.log(pd_op2))/10*pd_op1.shape[0]
                else:
                    joint = np.multiply(pd_op1, pd_op2)
                    joint_entropy += -np.sum(joint*np.log(joint))/10*joint.shape[0]

            else:
                if cross:
                    x1, _, _, _,_ = pi.act(stochastic, ob, sampled_op1)
                    x2, _, _, _, _ = pi.act(stochastic, ob, sampled_op2)
                    x1 = softmax(x1)
                    x2 = softmax(x2)
                    x1 = np.clip(x1,1e-20, 1.0)
                    x2 = np.clip(x2,1e-20, 1.0)
                    cum_entropy += -np.sum(x1*np.log(x2))/x1.shape[0]
                else:
                    x1_prob, x2_prob = pi.get_action_prob(stochastic, ob, sampled_op1, sampled_op2)
                    x1_prob = np.clip(x1_prob,1e-20, 1.0)
                    x2_prob = np.clip(x2_prob, 1e-20, 1.0)
                    joint = np.multiply(x1_prob, x2_prob)
                    joint_entropy += np.clip(-np.sum(joint*np.log(joint))/joint.shape[0], a_min=1e-20, a_max=None)

        cum_entropy = cum_entropy/len(combinations)
        joint_entropy = joint_entropy/ len(combinations)
        return cum_entropy*2 if cross else joint_entropy*4