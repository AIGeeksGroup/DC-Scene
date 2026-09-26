"""Object-aligned teacher forcing and annotation-free CoCa decoding."""
import torch
import torch.nn.functional as F

from models.captioner_dccv2.captioner import hungarian_matching, proposal_dimension_select
from models.captioner_dccv2.helper import Matcher


def context_for(objects, features, enc_xyz, query_xyz):
    near = torch.cdist(query_xyz, enc_xyz).topk(min(32, enc_xyz.shape[1]), largest=False).indices
    batch, queries, k = near.shape
    context = proposal_dimension_select(features, near.reshape(batch, -1))
    context = context.reshape(batch, queries, k, -1)
    return torch.cat([objects.unsqueeze(2), context], dim=2)


def pooled_logits(coca, text, image_tokens):
    _, tokens = coca.embed_text(text)
    for attention, cross_attention in coca.multimodal_layers:
        tokens = cross_attention(attention(tokens), image_tokens)
    return coca.to_logits(tokens)


def caption_forward(model, outputs, inputs, objects, features, enc_xyz, query_xyz, is_eval):
    context = context_for(objects, features, enc_xyz, query_xyz)
    batch, queries = context.shape[:2]
    coca = model.coca
    if is_eval:
        captions = []
        # Pool the visual context once; generate causally without any GT caption.
        flat_context = context.flatten(0, 1)
        for chunk in flat_context.split(16):
            _, image_tokens = coca.embed_image(image_tokens=chunk)
            ids = torch.full((len(chunk), 1), model.tokenizer.bos_token_id,
                             device=chunk.device, dtype=torch.long)
            finished = torch.zeros(len(chunk), device=chunk.device, dtype=torch.bool)
            for _ in range(model.max_des_len):
                next_id = pooled_logits(coca, ids, image_tokens)[:, -1].argmax(-1)
                next_id = torch.where(finished, model.tokenizer.eos_token_id, next_id)
                ids = torch.cat([ids, next_id[:, None]], dim=1)
                finished |= next_id == model.tokenizer.eos_token_id
                if finished.all():
                    break
            captions.extend(model.tokenizer.batch_decode(ids[:, 1:].tolist()))
        outputs["lang_cap"] = [["sos " + captions[b * queries + q] + " eos"
                                for q in range(queries)] for b in range(batch)]
        return outputs

    matcher = Matcher(cost_class=1, cost_objectness=0, cost_giou=2, cost_center=0)
    assignments = hungarian_matching(matcher, outputs, inputs)
    target = proposal_dimension_select(inputs["reference_tokens"], assignments["per_prop_gt_inds"].long())
    masks = proposal_dimension_select(inputs["reference_masks"], assignments["per_prop_gt_inds"].long())
    selected = assignments["proposal_matched_mask"].bool() & masks.any(-1)
    rows = selected.nonzero()[:, 0]
    if len(rows) != batch or not torch.equal(rows, torch.arange(batch, device=rows.device)):
        raise ValueError("CoCa pair training/scoring requires one matched caption per scene occurrence")
    labels, mask = target[selected].long(), masks[selected]
    bos = torch.full_like(labels[:, :1], model.tokenizer.bos_token_id)
    shifted = torch.cat([bos, labels[:, :-1]], dim=1)
    image_embeds, image_tokens = coca.embed_image(image_tokens=context[selected])
    logits = pooled_logits(coca, shifted, image_tokens)
    # Equation (3): summed token NLL, with padding excluded and EOS included.
    token_loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none") * mask
    outputs["caption_nll"] = token_loss.sum(-1)
    outputs["caption_count"] = torch.ones(batch, device=labels.device)
    text_embeds, _ = coca.embed_text(labels)
    text_latents = coca.text_to_latents(text_embeds)
    scene_latents = coca.img_to_latents(image_embeds)
    outputs["scene_embeddings"] = scene_latents
    outputs["text_embeddings"] = text_latents
    similarity = scene_latents @ text_latents.T * coca.temperature.exp().clamp(max=100)
    # Captions of the same scene/object in a batch are positives, not false negatives.
    object_ids = inputs["gt_box_object_ids"].gather(1, inputs["target_box_index"][:, None]).squeeze(1)
    positive = (inputs["scan_idx"][:, None] == inputs["scan_idx"][None, :]) & (object_ids[:, None] == object_ids[None, :])
    labels_contrastive = positive.float() / positive.sum(-1, keepdim=True)
    contrastive = -(labels_contrastive * F.log_softmax(similarity, -1)).sum(-1).mean()
    contrastive += -(labels_contrastive * F.log_softmax(similarity.T, -1)).sum(-1).mean()
    outputs["loss"] = outputs["loss"] + token_loss.sum() / mask.sum().clamp_min(1) + contrastive / 2
    return outputs
