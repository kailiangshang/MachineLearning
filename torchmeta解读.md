官方代码地址：tristandeleu/pytorch-meta: A collection of extensions and data-loaders for few-shot learning & meta-learning in PyTorch (github.com)
	版本信息：1.8.0（下载时间节点：2023/4/16）
	环境要求（其他环境部分要求可查看setup.py文件）：
	Python 3.6 or above
	PyTorch 1.4 or above
	Torchvision 0.5 or above
	
	代码整体更加关注数据管道的工作，针对具体的模型部分：例如MAML，Reptile等只给出了示例的方法，因此以下调研更多的关注于其data的pipeline实现，并实现自用。
	主要包含部分为：
	1）MetaDataset()，定义元数据集的基类。
	2）ClassDataset()，定义类别数据集的基类。
	3）CombinationMetaDataset() 实现将原始的数据集转换为可操作的元数据集形式。
	4）MetaDataLoader() 实现元数据集加载。
	5）BatchMetaDataLoader() 实现元批次数据集加载。
	6）上述中依赖模块解释。
	以下分别展开进行概述。
 	1.	MetaDataset元数据集的基类
	继承：无继承
	重要参数：
	meta_train=False, meta_val=False, meta_test=False,meta_split=None
	均为指定的划分模式，注意每次只能有一种模式传入。
	np_random：对np.random.RandomState(seed=seed)的实例化，方便后续继承该类的子类实现任务中类别的采样。
	其他参数：target_transform详见torchvison.transforms，dataset_transform详见torchmeta.transforms，这里不再叙述。
	重要类内函数：
	__getitem__：继承该基类时需要实现的方法。使得类实例化后可以通过索引的方式取值。
	__len__：继承该基类需要实现的方法。与getitem方法对应，给定实例化后的数据长度。
	__iter__：使得该基类称为一个生成器，也就是可迭代对象，可通过重复使用next(instance)实现对数据的调用。
	sample_task：采样，实例化后对象的随机选择数据。且可以重复调用该方法。具体细节详见python官网 return self的用法。
	其他函数：seed实例化np.random.RandomState

	2.  ClassDataset，定义类别数据集的基类。
	继承：无。
	重要参数：
	meta_train=False, meta_val=False, meta_test=False,meta_split=None
		均为指定的划分模式，注意每次只能有一种模式传入。
	class_augmentations：使用函数扩充现有类别，这些类是对已有类的转换。
	重要类内函数：
	__getitem__：继承该类时需要实现的方法。
	num_classes：通过property装饰，将该方法变为类属性，同样继承该类时需要实现该方法。表示类别的数量。
	__len__：返回类别总数量。


	3.  CombinationMetaDataset，组合元数据集基类。
	继承：MetaDataset
	重要参数：
	dataset：包含所有类别数据的数据集合。注意：这里的dataset为ClassDataset实例。
	num_classes_per_task：每个任务集合的类别数量。也就是所谓的N-way.
	其他参数：target_transform详见torchvison.transforms，dataset_transform详见torchmeta.transforms，这里不再叙述。
	注：其他初始化参数参照MetaDataset
	重要类内函数：
	__getitem__：外部实例化对象可以通过索引得到单个的元任务数据集。注意：这里的索引传入必须为int组合成的元组，元组中的每个int表示取到的是哪一个类别。依赖模块：torchmeta.utils.data.task.ConcatTask
	__len__：总共可以采样的不同分类任务的数目。C_n^k实现。其中n表示总类别数目，k表示每次抽取的类别数目。注意：这里产生的任务集合中的类别组合都是唯一的。
	__iter__：使用itertools.combinations函数对所有的类别进行排列组合。注意：官方在这里的实现在运行上是错误的！
	sample_task：任务采样函数的重写，利用父类中的参数np_random.choice实现类别的采样，且每次采样都生成一个代表类别的array对象并转换为元组，最后调用自身的getitem方法，将元组传入，得到一个任务集合。
	重要实现：外部通过传入dataset（注意是ClassDataset的实例，其次该实例中应该包含所有的类别的所有数据），初始化每个任务中的类别数目完成对象实例化。然后实例通过调用sample_tesk的方法，可以实现对一个任务集合的采样，采样依赖类中getitem方法的实现（该方法同时依赖ConcatTask模块）。而整个实例的长度为所有可能的任务集合的排列组合总数。
	4.  MetaDataLoader，元数据集的加载。
	继承：torch.utils.data.DataLoader
	重要参数（以下仅仅展示几个重要的初始化参数，其他参数请参照pytorch中的DataLoader参数）：
	dataset: 符合运行的数据集dataset，实际应为CombinationMetaDataset
	batch_size：1. 注意：这里的batch_size必须为1，具体原因请查看官方文档。
	shuffle：是否对加载的数据进行打乱操作。
	sampler：采样策略。
	其他参数集合：batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None
	重要实现：MetaDataLoader通过初始化DataLoader中的采样策略来实现元数据集的采样。参数sampler在该类中进行了重要操作，根据是否打乱数据生成了两个采样策略：1）如果shuffle为True，那么使用模块CombinationRandomSampler生成采样策略，并初始化DataLoader。2）如果shuffle为False，那么使用模块CombinationSequentialSampler，生成采样策略。执行完毕后，将shuffle置为Fasle。
	将采样策略传递给DataLoader进行初始化，得到单个的元数据集。
	注：采样策略在DataLoader中如何实现，请参照官方文档。
		 模块的解读请参照最后部分，模块解读。
	
	5.  BatchMetaDataLoader，元批次数据集加载。
	4中只针对单个元数据集加载进行了操作这里为了实现元批次数据集加载，进行了封装。
	继承：MetaDataLoader
	重要参数：
	无。注：这里的其他参数同MetaDataloader，仅仅对DataLoader的collate_fn参数进行了初始化，且外部不可以传入，而是通过模块BatchMetaCollate实现。
	重要实现：通过对MetaDataLoader的继承，可以实现一个元数据集的形成和加载，为了生成元批次数据集，通过collate_fn参数，可以自定义取出一个batch数据的格式。
  
  
	6.  上述中依赖模块部分解读
	torchmeta.utils.data.task.ConcatTask
	为了与torch兼容，实现以下类的继承。
	从torch.utils.data导入Dataset（源码中将此处另命名为Dataset_），ConcatDataset，Subset三个基类。
	1）自定义类Dataset，继承torch的Dataset给定参数index以及transform相关操作。
	2）自定义类Task，继承1）中Dataset，表示一个分类任务的基类。增加参数num_classes，表示该分类任务的类别数量，为int类型。其他参数来自父类。
	3）自定义ConcatTask类，父类包括：Task和ConcatDataset（来自torch）。需要给定参数datasets， num_classes，以及相关的变换操作。内部将初始化的index和num_classes传递给Task初始化，将datasets传递给ConcatDataset初始化，注意这里的datasets为一个列表，列表中存放着各类数据集。这里的getitem继承了ConcatDataset的方法（具体请查看torch官方的实现）。
	依赖该模块后的CombinationMetaDataset的实现流程：
	1）初始化所有的类别数据集，并将其存放至列表当中：[[cls0_0, cls0_1,  … ], [cls1_0, cls1_1,  …], …],初始化每个任务中的类别数目num_classes_per_task，通过实例对象[(idx0, idx1, .. , idxn)]的调用方式，将元组传递给CombinationMetaDataset.__getitem__， 该方法将元组解包，得到一个任务集（每个idx为一个类别），通过解包的索引值得到本次任务集的数据集合并打包成列表。
	2）将1）中的数据集列表和num_classes_per_task（由ConcatTask中的num_classes参数接收）传入ConcatTask模块，对数据集列表中的数据进行拼接，拼接完成后返回该任务集合。

	CombinationRandomSampler和CombinationSequentialSampler模块
	两者均有定义iter方法，因此外部实例化是可以将通过iter(instance)实现将两者转换为生成器。
	两者实现的逻辑相同，唯一一点不同在于前者继承官方的torch.utils.data.sampler.SequentialSampler， 后者则为RandomSampler。
	初始化参数data_source，应为CombinationMetaDataset实例，调用len方法获取data_source.data_set的所有类别数目，并获取每个任务中的类别数目，然后采用排列组合和随机排列组合的方式得到所有的可能的任务集合。
	依赖关系下的MetaDataLoader实现：
	实例化CombinationMetaDataset形成dataset传入MetaDataLoader，使用上述模块生成采样策略，将采样策略传递给父类DataLoader（torch官方的DataLoader），完成元数据集的加载。
	而针对BatchMetaDataLoader，元批次数据集加载，是通过BatchMetaCollate生成自定义取出batch数据集的格式，实现元批次数据集的加载。
torchmeta使用案例
	使用案例，以torchmeta官方的omniglot为例
	实例化类omniglot，并将实例化对象传入BatchMetaDataLoader中返回一个数据生成器对象。
1）定义OmniglotDataset类，继承torchmeta的Dataset类，注意：不是torch的Dataset类。通过索引取到的是一个一个的数据以及数据对应的索引，为一个元组对象。(image, target)
2）定义OmniglotClassDataset类，继承torchmeta的ClassDataset类，类中的几个重要属性：data表示所有的数据 labels表示所有的标签，num_classes表示所有的类别数量。
3）定义Omniglot类，继承torchmeta的CombinationMetaDataset类，参数解释同CombinationMetaDataset。
	实现流程：（具体的参数调用实例化参照torchmeta的官方文档）
	 
![image](https://user-images.githubusercontent.com/103066922/232499649-6c31d6ac-0496-4fd6-84c5-f33674971e25.png)

