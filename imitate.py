from imitation_builder import ImitationBuilder_Stabilization, ImitationBuilder_Wall, ImitationBuilder_Wall2, ImitationBuilder_Wall3

if __name__ == "__main__":
    EVALUATE_EXPERT = False
    TRAIN_POLICY = True

    # imitator, env, controller, safety_filter = ImitationBuilder_Stabilization().build()
    # imitator, env, controller, safety_filter = ImitationBuilder_Wall().build()
    # imitator, env, controller, safety_filter = ImitationBuilder_Wall2().build()
    imitator, env, controller, safety_filter = ImitationBuilder_Wall3().build()

    # evaluate expert to get an idea of the reward achievable
    if EVALUATE_EXPERT:
        imitator.evaluate_expert(n_eps=1)

    # train network to imitate expert
    if TRAIN_POLICY:
        imitator.train()
