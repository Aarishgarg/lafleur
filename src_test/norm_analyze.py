import torch
from nemo.collections.asr.models import EncDecRNNTModel


class LayerNormMonitor:
    def __init__(self, model, target_layer_type=torch.nn.LayerNorm):
        self.hooks = []
        # self.stats 現在包含 Tensor (max_dim_tensor) 和 Scalar Tensors (global metrics)
        self.stats = {} 
        self.model = model
        self.target_layer_type = target_layer_type

    def register(self):
        print("正在註冊 LayerNorm Hooks...")
        for name, module in self.model.named_modules():
            if isinstance(module, self.target_layer_type):
                self.hooks.append(module.register_forward_hook(self._get_hook(name)))

    def _get_hook(self, name):
        def hook(model, input, output):
            # output shape: [Batch, Time, Dim]
            with torch.no_grad():
                
                # 1. Max per Dimension (for plot)
                print(output.shape)
                current_max_per_dim = output.detach().abs().amax(dim=(0, 1)).float()
                
                # 2. Scalar Stats 
                current_global_abs_min = output.detach().abs().min().float()
                current_global_abs_max = output.detach().abs().max().float()
                current_global_abs_mean = output.detach().abs().mean().float()
                current_global_std = output.detach().std().float() 
                
                if name not in self.stats:
                    self.stats[name] = {
                        'max_dim_tensor': current_max_per_dim, # Tensor for plot
                        'global_abs_min': current_global_abs_min,  # CUDA Scalar Tensor
                        'global_abs_max': current_global_abs_max,  # CUDA Scalar Tensor
                        'global_abs_mean': current_global_abs_mean,
                        'global_std': current_global_std,
                        'count': 1
                    }
                else:
                    self.stats[name]['max_dim_tensor'] = torch.maximum(self.stats[name]['max_dim_tensor'], current_max_per_dim)
                    self.stats[name]['global_abs_max'] = torch.maximum(self.stats[name]['global_abs_max'], current_global_abs_max)
                    self.stats[name]['global_abs_min'] = torch.maximum(self.stats[name]['global_abs_min'], current_global_abs_max)
                    
                    count = self.stats[name]['count']
                    self.stats[name]['global_abs_mean'] = (self.stats[name]['global_abs_mean'] * count + current_global_abs_mean) / (count + 1)
                    self.stats[name]['global_std'] = (self.stats[name]['global_std'] * count + current_global_std) / (count + 1)
                    self.stats[name]['count'] += 1
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        print("Hooks 已移除。")
        
    
    def print_layer_statistics(self, threshold=20.0):
        if not self.stats:
            print("沒有收集到數據。")
            return

        print("\n" + "="*100)
        print("Detailed LayerNorm Activation Statistics")
        print("="*100)

        data_for_table = self.stats.items()
        grouped_stats = {} # { 'FF': [ (short_name, stats), ... ], 'Conv': [...], ... }
        
        for name, stats in data_for_table:
            min_abs = stats['global_abs_min'].cpu().item()
            max_abs = stats['global_abs_max'].cpu().item()
            mean_abs = stats['global_abs_mean'].cpu().item()
            std_dev = stats['global_std'].cpu().item()
            max_dim_val = stats['max_dim_tensor'].max().cpu().item()
            is_spike = "⚠️ SPIKE" if max_dim_val > threshold else ("DEAD?" if std_dev < 0.1 else "OK")


            parts = name.split('.')
            try:
                layer_index = parts[parts.index('layers') + 1]
                module_name_raw = parts[-1]
                
                if 'feed_forward1' in module_name_raw or 'feed_forward2' in module_name_raw:
                    module_type = 'FeedForward (FF)'
                    module_name = 'FF' + module_name_raw[-1]
                elif 'conv' in module_name_raw:
                    module_type = 'Convolution (Conv)'
                    module_name = 'Conv'
                elif 'att' in module_name_raw:
                    module_type = 'Attention (Att)'
                    module_name = 'Att'
                elif 'out' in module_name_raw:
                    module_type = 'Output (Out)'
                    module_name = 'Out'
                else:
                    module_type = 'Other'
                    module_name = module_name_raw
                
                short_name = f'L{layer_index}-{module_name}'
            except:
                module_type = 'Other'
                short_name = name.split('.')[-1]
            

            if module_type not in grouped_stats:
                grouped_stats[module_type] = []
            
            grouped_stats[module_type].append({
                'short_name': short_name,
                'min_abs': min_abs,
                'max_abs': max_abs,
                'mean_abs': mean_abs,
                'std_dev': std_dev,
                'max_dim_val': max_dim_val,
                'is_spike': is_spike
            })

        for module_type, stats_list in grouped_stats.items():
            print(f"\n--- LayerNorm Statistics for Type: {module_type} ---")
            
            # Prepare header
            header = ["Layer Index", "Max Abs", "Min Abs", "Mean Abs", "Std Dev", "Max Per Dim", "Spike Check"]
            print(f"{header[0]:<15} | {header[1]:>10} | {header[2]:>10} | {header[3]:>10} | {header[4]:>12} | {header[5]}")
            print("-" * 75)
            
            for item in stats_list:
                print(f"{item['short_name']:<15} | {item['max_abs']:>10.4f} | {item['min_abs']:>10.7f} | {item['mean_abs']:>10.4f} | {item['std_dev']:>10.4f} | {item['max_dim_val']:>12.4f} | {item['is_spike']}")
        
        print("\n" + "="*100)
        print(f"診斷參考：\n1. SPIKE: 'Max Per Dim' > {threshold}，特徵爆炸。\n2. DEAD?: 'Std Dev' < 0.1，模組可能不活躍或死亡。")
        
    

manifest = '/ws/code/ASR/parakeet/src/unit2.json'
asr_model = EncDecRNNTModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3', strict = False)
asr_model = asr_model.to("cuda")
asr_model.eval()
print(asr_model)

monitor = LayerNormMonitor(asr_model)
monitor.register()

total = []
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    with torch.no_grad():
        
        hypotheses = asr_model.transcribe(manifest, batch_size=64)
            

monitor.remove()
monitor.print_layer_statistics(threshold=20.0)