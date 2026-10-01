# arm-report-revision2

三条件V/SAC/script分别给计划n、有效n、删失n、未运行n、严格成功k/n、Wilson95%；n=0写NA。固定分母不因删失缩小。200/600保存首次成功步数；SAC仅官方200，不外推600。按类别×高度与场景列k/n，明确场景内相关，Wilson仅未校正描述。报告脚本600成功子集的V；不含P与配对项。

漏斗保持：check_path接受、TCP到局部AABB<=0.05m、is_grasped、同一步抓住且目标升高>=0.05m、官方五项同时成立。原始标志/累积交集分列，严格成功直接取官方五项，SAC的check_path层NA。

每次locate_point单列case/turn/step/observation-id/valid/目标碰撞三角形表面距离m，缺点为NA；仅评估，不进反馈/history/请求，不单凭距离断言模型失败。逐例保留note/hindsight与实测行动，区分错误信念、执行偏离、环境挫败、无法判定。

脚本至少4/5且5例有效才可提出测试组提案；不足即停止，不修补。记录五项判据、累计力、抓空/重抓、工具调用、费用及媒体精确文件/日期/结果/哈希。执行器CPU、脚本、Astra V分开报告。
