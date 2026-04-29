"""
图索引模块
实现实体和关系的键值对结构 (K,V)
K: 索引键（简短词汇或短语）
V: 详细描述段落（包含相关文本片段）
"""

# 实现了一个 图索引模块 ，主要功能是将Neo4j图数据库中的实体和关系转换为键值对（K,V）结构，用于后续的RAG检索系统

import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

@dataclass
class EntityKeyValue:  # 存储图数据库中实体的键值对表示，每个实体有一个或多个索引键，包含实体的详细描述内容，记录实体类型和元数据
    entity_name: str
    index_keys: List[str]  # 索引键列表
    value_content: str     # 详细描述内容
    entity_type: str       # 实体类型 (Recipe, Ingredient, CookingStep)
    metadata: Dict[str, Any]

@dataclass 
class RelationKeyValue:  # 存储图数据库中关系的键值对表示， 每个关系有多个索引键，包括全局主题，包含关系的详细描述内容，记录关系类型、源实体和目标实体
    """关系键值对"""
    relation_id: str
    index_keys: List[str]  # 多个索引键（可包含全局主题）
    value_content: str     # 关系描述内容
    relation_type: str     # 关系类型
    source_entity: str     # 源实体
    target_entity: str     # 目标实体
    metadata: Dict[str, Any]

class GraphIndexingModule:
    """
    图索引模块
    核心功能：
    1. 为实体创建键值对（名称作为唯一索引键）
    2. 为关系创建键值对（多个索引键，包含全局主题）
    3. 去重和优化图操作
    4. 支持增量更新
    """
    
    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        
        # 键值对存储
        self.entity_kv_store: Dict[str, EntityKeyValue] = {}
        self.relation_kv_store: Dict[str, RelationKeyValue] = {}
        
        # 索引映射：key -> entity/relation IDs
        self.key_to_entities: Dict[str, List[str]] = defaultdict(list)
        self.key_to_relations: Dict[str, List[str]] = defaultdict(list)
        
        # 假设有一个 "宫保鸡丁" 菜谱系统，包含以下数据
        # entity_kv_store = {
        #     # 菜谱实体
        #     "recipe_001": EntityKeyValue(
        #         entity_name="宫保鸡丁",
        #         index_keys=["宫保鸡丁"],
        #         value_content="""菜品名称: 宫保鸡丁
        #         描述: 经典川菜，酸甜微辣
        #         分类: 热菜
        #         菜系: 川菜
        #         难度: 中等
        #         制作时间: 30分钟""",
        #         entity_type="Recipe",
        #         metadata={"node_id": "recipe_001", "properties": {...}}
        #     ),
            
        #     # 食材实体
        #     "ing_001": EntityKeyValue(
        #         entity_name="鸡胸肉",
        #         index_keys=["鸡胸肉"],
        #         value_content="""食材名称: 鸡胸肉
        #         类别: 肉类
        #         营养信息: 高蛋白低脂肪
        #         储存方式: 冷藏保存""",
        #         entity_type="Ingredient",
        #         metadata={"node_id": "ing_001", "properties": {...}}
        #     ),
            
        #     "ing_002": EntityKeyValue(
        #         entity_name="花生米",
        #         index_keys=["花生米"],
        #         value_content="""食材名称: 花生米
        #         类别: 坚果类
        #         营养信息: 富含蛋白质和脂肪""",
        #         entity_type="Ingredient",
        #         metadata={"node_id": "ing_002", "properties": {...}}
        #     ),
            
        #     # 烹饪步骤实体
        #     "step_001": EntityKeyValue(
        #         entity_name="步骤_step_001",
        #         index_keys=["步骤_step_001"],
        #         value_content="""烹饪步骤: 步骤_step_001
        #         步骤描述: 将鸡胸肉切丁，加入料酒、盐、淀粉腌制15分钟
        #         步骤顺序: 1
        #         技巧: 腌制时加入少量蛋清可使肉质更嫩""",
        #         entity_type="CookingStep",
        #         metadata={"node_id": "step_001", "properties": {...}}
        #     ),
        # }

        #     relation_kv_store = {
        #     # 菜谱-食材关系（宫保鸡丁需要鸡胸肉）
        #     "rel_0_recipe_001_ing_001": RelationKeyValue(
        #         relation_id="rel_0_recipe_001_ing_001",
        #         index_keys=["REQUIRES", "食材搭配", "烹饪原料", "宫保鸡丁_食材", "鸡胸肉"],
        #         value_content="""关系类型: REQUIRES
        #         源实体: 宫保鸡丁
        #         目标实体: 鸡胸肉""",
        #         relation_type="REQUIRES",
        #         source_entity="recipe_001",
        #         target_entity="ing_001",
        #         metadata={"source_name": "宫保鸡丁", "target_name": "鸡胸肉", ...}
        #     ),
            
        #     # 菜谱-食材关系（宫保鸡丁需要花生米）
        #     "rel_1_recipe_001_ing_002": RelationKeyValue(
        #         relation_id="rel_1_recipe_001_ing_002",
        #         index_keys=["REQUIRES", "食材搭配", "烹饪原料", "宫保鸡丁_食材", "花生米"],
        #         value_content="""关系类型: REQUIRES
        #         源实体: 宫保鸡丁
        #         目标实体: 花生米""",
        #         relation_type="REQUIRES",
        #         source_entity="recipe_001",
        #         target_entity="ing_002",
        #         metadata={"source_name": "宫保鸡丁", "target_name": "花生米", ...}
        #     ),
            
        #     # 菜谱-步骤关系
        #     "rel_2_recipe_001_step_001": RelationKeyValue(
        #         relation_id="rel_2_recipe_001_step_001",
        #         index_keys=["HAS_STEP", "制作步骤", "烹饪过程", "宫保鸡丁_步骤", "制作方法"],
        #         value_content="""关系类型: HAS_STEP
        #         源实体: 宫保鸡丁
        #         目标实体: 步骤_step_001""",
        #         relation_type="HAS_STEP",
        #         source_entity="recipe_001",
        #         target_entity="step_001",
        #         metadata={"source_name": "宫保鸡丁", "target_name": "步骤_step_001", ...}
        #     ),
        # }

        # key_to_entities = {
        #     "宫保鸡丁": ["recipe_001"],
        #     "鸡胸肉": ["ing_001"],
        #     "花生米": ["ing_002"],
        #     "步骤_step_001": ["step_001"],
            
        #     # 假设还有另一个菜谱也用到鸡胸肉
        #     # "鸡胸肉": ["ing_001", "ing_005"],  # 多个实体共享同一名称
        # }

        # key_to_relations = {
        #     # 按关系类型索引
        #     "REQUIRES": ["rel_0_recipe_001_ing_001", "rel_1_recipe_001_ing_002"],
        #     "HAS_STEP": ["rel_2_recipe_001_step_001"],
            
        #     # 按主题索引
        #     "食材搭配": ["rel_0_recipe_001_ing_001", "rel_1_recipe_001_ing_002"],
        #     "烹饪原料": ["rel_0_recipe_001_ing_001", "rel_1_recipe_001_ing_002"],
        #     "制作步骤": ["rel_2_recipe_001_step_001"],
        #     "烹饪过程": ["rel_2_recipe_001_step_001"],
            
        #     # 按菜谱名称索引
        #     "宫保鸡丁_食材": ["rel_0_recipe_001_ing_001", "rel_1_recipe_001_ing_002"],
        #     "宫保鸡丁_步骤": ["rel_2_recipe_001_step_001"],
            
        #     # 按食材名称索引
        #     "鸡胸肉": ["rel_0_recipe_001_ing_001"],
        #     "花生米": ["rel_1_recipe_001_ing_002"],
        # }

    # 将Recipe、Ingredient和CookingStep实体转换为键值对，为每个实体生成索引键（使用实体名称），构建实体的详细描述内容，建立索引键到实体的映射
    def create_entity_key_values(self, recipes: List[Any], ingredients: List[Any], 
                                cooking_steps: List[Any]) -> Dict[str, EntityKeyValue]:
        """
        为实体创建键值对结构
        每个实体使用其名称作为唯一索引键
        """
        logger.info("开始创建实体键值对...")
        
        # 处理菜谱实体
        for recipe in recipes:
            entity_id = recipe.node_id
            entity_name = recipe.name or f"菜谱_{entity_id}"
            
            # 构建详细内容
            content_parts = [f"菜品名称: {entity_name}"]
            
            if hasattr(recipe, 'properties'):
                props = recipe.properties
                if props.get('description'):
                    content_parts.append(f"描述: {props['description']}")
                if props.get('category'):
                    content_parts.append(f"分类: {props['category']}")
                if props.get('cuisineType'):
                    content_parts.append(f"菜系: {props['cuisineType']}")
                if props.get('difficulty'):
                    content_parts.append(f"难度: {props['difficulty']}")
                if props.get('cookingTime'):
                    content_parts.append(f"制作时间: {props['cookingTime']}")
            
            # 创建键值对
            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],  # 使用名称作为唯一索引键，这里提前保存下键，之后去重后会使用这里存储的重建映射
                value_content='\n'.join(content_parts),
                entity_type="Recipe",
                metadata={
                    "node_id": entity_id,
                    "properties": getattr(recipe, 'properties', {})
                }
            )
            
            self.entity_kv_store[entity_id] = entity_kv  # 菜谱id -> 菜谱详细信息
            self.key_to_entities[entity_name].append(entity_id)  # 菜谱名 -> 菜谱id
        
        # 处理食材实体
        for ingredient in ingredients:
            entity_id = ingredient.node_id
            entity_name = ingredient.name or f"食材_{entity_id}"
            
            content_parts = [f"食材名称: {entity_name}"]
            
            if hasattr(ingredient, 'properties'):
                props = ingredient.properties
                if props.get('category'):
                    content_parts.append(f"类别: {props['category']}")
                if props.get('nutrition'):
                    content_parts.append(f"营养信息: {props['nutrition']}")
                if props.get('storage'):
                    content_parts.append(f"储存方式: {props['storage']}")
            
            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],
                value_content='\n'.join(content_parts),
                entity_type="Ingredient",
                metadata={
                    "node_id": entity_id, 
                    "properties": getattr(ingredient, 'properties', {})  
                }
            )
            
            self.entity_kv_store[entity_id] = entity_kv  # 食材id -> 食材详细信息
            self.key_to_entities[entity_name].append(entity_id)  # 食材名 -> 食材id
        
        # 处理烹饪步骤实体
        for step in cooking_steps:
            entity_id = step.node_id
            entity_name = f"步骤_{entity_id}"
            
            content_parts = [f"烹饪步骤: {entity_name}"]
            
            if hasattr(step, 'properties'):
                props = step.properties
                if props.get('description'):
                    content_parts.append(f"步骤描述: {props['description']}")
                if props.get('order'):
                    content_parts.append(f"步骤顺序: {props['order']}")
                if props.get('technique'):
                    content_parts.append(f"技巧: {props['technique']}")
                if props.get('time'):
                    content_parts.append(f"时间: {props['time']}")
            
            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],
                value_content='\n'.join(content_parts),
                entity_type="CookingStep", 
                metadata={
                    "node_id": entity_id,
                    "properties": getattr(step, 'properties', {})
                }
            )
            
            self.entity_kv_store[entity_id] = entity_kv  # 步骤id -> 步骤详细信息
            self.key_to_entities[entity_name].append(entity_id)  # 步骤名 -> 步骤id
        
        logger.info(f"实体键值对创建完成，共 {len(self.entity_kv_store)} 个实体")
        return self.entity_kv_store
    
    def create_relation_key_values(self, relationships: List[Tuple[str, str, str]]) -> Dict[str, RelationKeyValue]:
        """
        为关系创建键值对结构
        关系可能有多个索引键，包含从LLM增强的全局主题
        """
        logger.info("开始创建关系键值对...")
        
        for i, (source_id, relation_type, target_id) in enumerate(relationships):
            relation_id = f"rel_{i}_{source_id}_{target_id}"
            
            # 获取源实体和目标实体信息
            source_entity = self.entity_kv_store.get(source_id)
            target_entity = self.entity_kv_store.get(target_id)
            
            if not source_entity or not target_entity:
                continue
            
            # 构建关系描述
            content_parts = [
                f"关系类型: {relation_type}",
                f"源实体: {source_entity.entity_name} ({source_entity.entity_type})",
                f"目标实体: {target_entity.entity_name} ({target_entity.entity_type})"
            ]
            
            # 生成多个索引键（包含全局主题）
            index_keys = self._generate_relation_index_keys(
                source_entity, target_entity, relation_type
            )
            
            # 创建关系键值对
            relation_kv = RelationKeyValue(
                relation_id=relation_id,
                index_keys=index_keys,  # 这里提前保存下键，之后去重后会使用这里存储的重建映射
                value_content='\n'.join(content_parts),
                relation_type=relation_type,
                source_entity=source_id,
                target_entity=target_id,
                metadata={
                    "source_name": source_entity.entity_name,
                    "target_name": target_entity.entity_name,
                    "created_from_graph": True
                }
            )
            
            self.relation_kv_store[relation_id] = relation_kv
            
            # 为每个索引键建立映射
            for key in index_keys:
                self.key_to_relations[key].append(relation_id)
        
        logger.info(f"关系键值对创建完成，共 {len(self.relation_kv_store)} 个关系")
        return self.relation_kv_store
    
    def _generate_relation_index_keys(self, source_entity: EntityKeyValue, 
                                    target_entity: EntityKeyValue, 
                                    relation_type: str) -> List[str]:
        """
        为关系生成多个索引键，包含全局主题
        """
        keys = [relation_type]  # 基础关系类型键
        
        # 根据关系类型和实体类型生成主题键
        if relation_type == "REQUIRES":
            # 菜谱-食材关系的主题键
            keys.extend([
                "食材搭配",
                "烹饪原料",
                f"{source_entity.entity_name}_食材",
                target_entity.entity_name
            ])
        elif relation_type == "HAS_STEP":
            # 菜谱-步骤关系的主题键
            keys.extend([
                "制作步骤",
                "烹饪过程",
                f"{source_entity.entity_name}_步骤",
                "制作方法"
            ])
        elif relation_type == "BELONGS_TO_CATEGORY":
            # 分类关系的主题键
            keys.extend([
                "菜品分类",
                "美食类别",
                target_entity.entity_name
            ])
        
        # 使用LLM增强关系索引键（可选）
        if getattr(self.config, 'enable_llm_relation_keys', False):
            enhanced_keys = self._llm_enhance_relation_keys(source_entity, target_entity, relation_type)
            keys.extend(enhanced_keys)
        
        # 去重并返回
        return list(set(keys))
    
    def _llm_enhance_relation_keys(self, source_entity: EntityKeyValue, 
                                 target_entity: EntityKeyValue, 
                                 relation_type: str) -> List[str]:
        """
        使用LLM增强关系索引键，生成全局主题
        """
        prompt = f"""
        分析以下实体关系，生成相关的主题关键词：
        
        源实体: {source_entity.entity_name} ({source_entity.entity_type})
        目标实体: {target_entity.entity_name} ({target_entity.entity_type})
        关系类型: {relation_type}
        
        请生成3-5个相关的主题关键词，用于索引和检索。
        返回JSON格式：{{"keywords": ["关键词1", "关键词2", "关键词3"]}}
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            return result.get("keywords", [])
            
        except Exception as e:
            logger.error(f"LLM增强关系索引键失败: {e}")
            return []
    
    def deduplicate_entities_and_relations(self):
        """
        去重相同的实体和关系，优化图操作
        """
        logger.info("开始去重实体和关系...")
        
        # 实体去重：基于名称（id互不相同，但是每个id的名字一定是唯一的，可能多个id对应同一个名字）
        name_to_entities = defaultdict(list)
        for entity_id, entity_kv in self.entity_kv_store.items():
            name_to_entities[entity_kv.entity_name].append(entity_id)
        
        # 合并重复实体
        entities_to_remove = []
        for name, entity_ids in name_to_entities.items():
            if len(entity_ids) > 1:
                # 保留第一个，合并其他的内容
                primary_id = entity_ids[0]
                primary_entity = self.entity_kv_store[primary_id]
                
                for entity_id in entity_ids[1:]:
                    duplicate_entity = self.entity_kv_store[entity_id]
                    # 合并内容
                    primary_entity.value_content += f"\n\n补充信息: {duplicate_entity.value_content}"
                    # 标记删除
                    entities_to_remove.append(entity_id)
        
        # 删除重复实体
        for entity_id in entities_to_remove:
            del self.entity_kv_store[entity_id]
        
        # 关系去重：基于源-目标-类型
        relation_signature_to_ids = defaultdict(list)
        for relation_id, relation_kv in self.relation_kv_store.items():
            signature = f"{relation_kv.source_entity}_{relation_kv.target_entity}_{relation_kv.relation_type}"
            relation_signature_to_ids[signature].append(relation_id)
        
        # 合并重复关系
        relations_to_remove = []
        for signature, relation_ids in relation_signature_to_ids.items():
            if len(relation_ids) > 1:
                # 保留第一个，删除其他
                for relation_id in relation_ids[1:]:
                    relations_to_remove.append(relation_id)
        
        # 删除重复关系
        for relation_id in relations_to_remove:
            del self.relation_kv_store[relation_id]
        
        # 重建索引映射
        self._rebuild_key_mappings()
        
        logger.info(f"去重完成 - 删除了 {len(entities_to_remove)} 个重复实体，{len(relations_to_remove)} 个重复关系")
    
    def _rebuild_key_mappings(self):
        """重建键到实体/关系的映射"""
        self.key_to_entities.clear()
        self.key_to_relations.clear()
        
        # 重建实体映射
        for entity_id, entity_kv in self.entity_kv_store.items():
            for key in entity_kv.index_keys:
                self.key_to_entities[key].append(entity_id)
        
        # 重建关系映射
        for relation_id, relation_kv in self.relation_kv_store.items():
            for key in relation_kv.index_keys:
                self.key_to_relations[key].append(relation_id)
    
    def get_entities_by_key(self, key: str) -> List[EntityKeyValue]:
        """根据索引键获取实体"""
        entity_ids = self.key_to_entities.get(key, [])
        return [self.entity_kv_store[eid] for eid in entity_ids if eid in self.entity_kv_store]
    
    def get_relations_by_key(self, key: str) -> List[RelationKeyValue]:
        """根据索引键获取关系"""
        relation_ids = self.key_to_relations.get(key, [])
        return [self.relation_kv_store[rid] for rid in relation_ids if rid in self.relation_kv_store]
    

    
    def get_statistics(self) -> Dict[str, Any]:
        """获取键值对存储统计信息"""
        return {
            "total_entities": len(self.entity_kv_store),
            "total_relations": len(self.relation_kv_store),
            "total_entity_keys": sum(len(kv.index_keys) for kv in self.entity_kv_store.values()),
            "total_relation_keys": sum(len(kv.index_keys) for kv in self.relation_kv_store.values()),
            "entity_types": {
                "Recipe": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "Recipe"]),
                "Ingredient": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "Ingredient"]),
                "CookingStep": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "CookingStep"])
            }
        } 