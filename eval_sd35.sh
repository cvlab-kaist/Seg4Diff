# Example code for evaluating Stable Diffusion 3.5 on ovss setting
sh eval_ovss.sh ./configs/eval_ovss.yaml 2 ./output/sd35 \
    MODEL.BACKBONE.ATTENTION_LAYERS "[$layer]" MODEL.BACKBONE.NAME SD35Backbone MODEL.BACKBONE.USE_LORA False MODEL.BACKBONE.KEEP_HEAD False MODEL.BACKBONE.SOFTMAX True \
    MODEL.WEIGHTS None

cat ./output/sd35/eval/log.txt | grep copypaste