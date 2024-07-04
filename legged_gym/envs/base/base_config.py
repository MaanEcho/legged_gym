import inspect

class BaseConfig:
    def __init__(self) -> None:
        """ Initializes all member classes recursively. Ignores all names starting with '__' (buit-in methods)."""
        """ 递归地初始化所有成员类。忽略所有名称以“__”开头的（内置方法）。"""
        self.init_member_classes(self)
    
    @staticmethod
    def init_member_classes(obj):
        # iterate over all attributes names
        # 迭代所有属性名称
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            # 忽视内置属性
            # 如果键以“__”开头，则跳过
            if key=="__class__":
                continue
            # get the corresponding attribute object
            # 获取相应的属性对象
            var =  getattr(obj, key)
            # check if the attribute is a class
            # 检查它是否是一个类
            if inspect.isclass(var):
                # instantate the class
                # 实例化类
                i_var = var()
                # set the attribute to the instance instead of the type
                # 将属性设置到实例上而不是类型上
                setattr(obj, key, i_var)
                # recursively init members of the attribute
                # 递归初始化属性的成员
                BaseConfig.init_member_classes(i_var)