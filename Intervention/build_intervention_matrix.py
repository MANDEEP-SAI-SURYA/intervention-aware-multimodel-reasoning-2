import torch


def build_intervention_matrix(model, image_path, question, objects, adjacency, device):
    """
    Computes causal effect proxy for each node:
    effect[i] = |f(x) - f(do(x_i))|
    """

    model.eval()

    with torch.no_grad():

        # factual prediction
        factual = model(image_path, question)["logits"]

        num_nodes = objects.shape[1]
        effects = []

        for i in range(num_nodes):

            # create intervention (remove node i)
            intervened_objects = objects.clone()
            intervened_objects[:, i, :] = 0.0

            # run model with intervention
            cf_output = model.forward_from_objects(
                intervened_objects,
                question
            )

            cf_logits = cf_output["logits"]

            # causal effect magnitude
            effect = torch.abs(factual - cf_logits).mean(dim=-1)

            effects.append(effect)

        return torch.stack(effects, dim=1)