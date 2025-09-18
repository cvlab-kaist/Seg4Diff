# Example code for evaluating Flux.1-dev on ovss setting
sh eval_ovss.sh ./configs/eval_ovss.yaml 2 ./output/flux \
    MODEL.BACKBONE.ATTENTION_LAYERS "[$layer]" MODEL.BACKBONE.NAME FluxBackbone MODEL.BACKBONE.USE_LORA False MODEL.BACKBONE.KEEP_HEAD False MODEL.BACKBONE.SOFTMAX True \
    MODEL.WEIGHTS None

cat ./output/flux/eval/log.txt | grep copypaste