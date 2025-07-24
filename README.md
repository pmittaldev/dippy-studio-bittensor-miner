# Dippy Studio Bittensor Miner

A specialized Bittensor subnet miner for AI model training, focusing on LoRA fine-tuning of diffusion models like FLUX.1-dev.

## Prerequisites

Before running this miner, you must:

1. **Set up a HuggingFace account and token**:
   - Create an account at [HuggingFace](https://huggingface.co)
   - Generate an access token at [HuggingFace Settings](https://huggingface.co/settings/tokens)
   - The token must have write permissions to create and upload models

2. **Accept FLUX.1-dev model license**:
   - Visit the [FLUX.1-dev model page](https://huggingface.co/black-forest-labs/FLUX.1-dev)
   - Read and accept the license agreement
   - Without accepting the license, the model download will fail

## Running the Miner

### Using Docker (Recommended)

Docker handles all dependencies and environment setup for you. No virtual environment needed!

1. Create a `.env` file with your HuggingFace token:
   ```bash
   echo "HF_TOKEN=your_huggingface_token_here" > .env
   ```

2. Start the miner:
   ```bash
   docker compose up -d
   ```

   This will:
   - Build the Docker image with all dependencies
   - Start the miner API on port 8091
   - Mount necessary volumes for model storage
   - Enable GPU access for training

3. Check logs:
   ```bash
   docker compose logs -f
   ```


## How It Works

The miner operates as a training service that:

1. **Receives training requests** via POST to `/train` endpoint with:
   - `job_type`: "lora_training"
   - `params`: Including prompt, image_b64, seed
   - `job_id`: Unique identifier
   - `validator_endpoint`: Callback URL for results

2. **Processes training jobs**:
   - Generates configuration from request parameters
   - Performs LoRA fine-tuning on FLUX.1-dev
   - Uses provided prompts and images for training

3. **Uploads trained models** to HuggingFace:
   - Creates a new repository for each training job
   - Uploads LoRA weights and metadata
   - Makes models publicly accessible

4. **Reports completion** back to the validator endpoint

## Manual Training

For testing or custom training jobs:

```bash
python run.py config/your_config.yaml --seed 42
```

## System Requirements

- **GPU**: NVIDIA GPU with at least 24GB VRAM (recommended: A100, H100)
- **CUDA**: Version 11.8 or higher
- **RAM**: 32GB minimum
- **Storage**: 100GB+ for model weights and training data
- **Docker**: Latest version with nvidia-container-toolkit

## Monitoring and Logs

- **API logs**: Check `docker compose logs -f`
- **Training progress**: Monitor individual job outputs in logs
- **GPU usage**: Use `nvidia-smi` to monitor GPU utilization

## Troubleshooting

### Model Download Issues
If you encounter permission errors downloading FLUX.1-dev:
1. Ensure you've accepted the model license on HuggingFace
2. Verify your HF_TOKEN is correctly set
3. Check that your token has read permissions

### GPU Memory Issues
If training fails with CUDA out of memory:
1. Reduce batch size in configuration
2. Enable gradient checkpointing
3. Use a GPU with more VRAM

### Connection Issues
If the miner can't connect to validators:
1. Check firewall settings for port 8091
2. Ensure Docker networking is properly configured
3. Verify validator endpoints are accessible

## Additional Resources

- [Bittensor Documentation](https://docs.bittensor.com)
- [FLUX.1 Model Card](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [LoRA Training Guide](https://huggingface.co/docs/peft/conceptual_guides/lora)

## Support

For issues and questions:
- Check existing issues in the repository
- Join the Bittensor Discord community
- Review validator documentation for integration details