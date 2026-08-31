import albumentations as albu

class AbstractTransformWrapper:
    def __init__(self, param_list):
        self.param_list = param_list
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __str__(self):
        return self.__class__.__name__[:-7] # 返回类名
    
    def __next__(self):
        if self.index < len(self.param_list):
            param = self.param_list[self.index]
            self.index += 1
            
            if param == 0:
                return param, None
            else:
                return self._get_transform(param)
        else:
            self.index = 0
            raise StopIteration
    
    def _get_transform(self, param):
        raise NotImplementedError
    
        
class GaussianBlurWrapper(AbstractTransformWrapper):
    def _get_transform(self, param):
        return param, albu.GaussianBlur(
            blur_limit=(param, param),
            always_apply=True,
            p=1.0
        )
        
class GaussianNoiseWrapper(AbstractTransformWrapper):
    def _get_transform(self, param):
        return param, albu.GaussNoise(
            var_limit=(param, param),
            always_apply=True,
            p=1.0
        )
        
class JpegCompressionWrapper(AbstractTransformWrapper):
    def _get_transform(self, param):
        return param, albu.JpegCompression(
            quality_lower = param-1,
            quality_upper = param,
            p=1.0
        )

class ColorJitterWrapper(AbstractTransformWrapper):
    def _get_transform(self, param):
        return param, albu.ColorJitter(
            brightness=param,
            contrast=param,
            saturation=param,
            hue=0,        # 保持为0，避免偏色太夸张
            always_apply=True,
            p=1.0
        )

class ResolutionChangeWrapper(AbstractTransformWrapper):
    def _get_transform(self, param):
        return param, albu.Downscale(
            scale_min=param,
            scale_max=param,
            always_apply=True,
            p=1.0
        )

class RotationWrapper(AbstractTransformWrapper):
    def _get_transform(self, param):
        return param, albu.Rotate(
            limit=(param, param),   # 固定旋转角度
            border_mode=0,          # 使用constant填充，避免黑边过多
            value=0,
            mask_value=0,
            always_apply=True,
            p=1.0
        )



if __name__ == "__main__":
    # 示例用法
    param_list = [90, 80, 0]  # 传入一个装有int的列表，代表需要遍历的参数数值
    wrapper = JpegCompressionWrapper(param_list)

    for transform in wrapper:
        print(transform)


    for transform in wrapper:
        print(transform)
        
    print(str(wrapper))
        
