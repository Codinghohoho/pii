"""13 类字段、21 条正则：扫描原文并返回候选，不负责替换。

与当前 Demo 使用相同规则，独立保存，不依赖 server.py 或父目录模块。
字段边界表用于阻止串字段；候选仍可能重复/重叠，由 pipeline.py 合并。
"""
import re
from datetime import datetime
from .config import FIELDS


# 字段边界表：只用于结束上一字段，不会为这些标签自动增加脱敏类型。
# 按最长标签优先匹配，避免“联系人姓名”被拆成“联系人”和“姓名”。
FIELD_BOUNDARY_LABELS = (
    '患者姓名', '患者本人姓名', '姓名',
    '性别', '年龄', '民族', '国籍', '婚姻状况', '婚姻', '职业', '工作单位',
    '家庭住址', '家庭地址', '联系地址', '居住地址', '户籍地址', '户口地址',
    '通讯地址', '现住址', '住址', '地址', '出生地', '籍贯', '出生日期',
    '身份证号码', '身份证件号', '身份证号', '身份证',
    '联系电话', '随访电话', '联系人电话', '电话号码', '手机号', '手机', '电话',
    '电子邮箱', '邮箱', '邮政编码', '邮编',
    '住院号', '病历号', '门诊号', '就诊号', '医保号',
    '申请单号', '检查申请号', '检查号', '报告编号', '床位号', '床号',
    '医院名称', '医院', '科室', '病区', '专业技术职称', '技术职称', '职称',
    '主治或主任医师签名', '上级医师签名', '住院医师签名', '医师签名',
    '医生姓名', '医师姓名', '医生', '随访负责人', '主治医师', '责任护士',
    '术者', '助手', '助理护士', '手术者签名', '手术者姓名', '记录签名', '主持人', '审核签名', '签名', '签字', '审批者',
    '联系人姓名', '代理人姓名', '紧急联系人', '联系人', '代理人', '与患者关系',
    '入院日期', '出院日期', '记录日期', '检查日期', '入院时间', '出院时间',
)
_FIELD_LABELS = '|'.join(re.escape(label) for label in sorted(FIELD_BOUNDARY_LABELS,key=len,reverse=True))
# 无冒号标签须从分隔处开始；标签后须有空白/行末，防止截断“民族路”等正文。
_FIELD_LEFT = r'(?<![^\s，,。；;：:|｜])'
_FIELD_NUMERIC = '年龄|住院号|病历号|门诊号|就诊号|医保号|申请单号|检查申请号|检查号|报告编号|床位号|床号|邮政编码|邮编'
_FIELD_BOUNDARY = re.compile(
    rf'(?:{_FIELD_LABELS})[ \t\u3000]*[：:]'
    rf'|{_FIELD_LEFT}(?:{_FIELD_LABELS})(?=[ \t\u3000\r\n]|\Z)'
    rf'|{_FIELD_LEFT}(?:{_FIELD_NUMERIC})(?=[0-9A-Za-z])'
    rf'|{_FIELD_LEFT}性别(?=[男女](?:[\s，,。；;]|\Z))'
)


def field_ranges(text):
    """按已知字段标签划分扫描范围。

    输入：``text`` 原文。
    输出：``[(start, end), ...]`` 范围列表，使用左闭右开字符坐标；不修改原文。
    """
    starts=sorted({0,*(m.start() for m in _FIELD_BOUNDARY.finditer(text))})
    return list(zip(starts,starts[1:]+[len(text)]))


def make_span(text, start, end, kind, source, detail=''):
    """创建统一格式的候选实体。

    输入：``text`` 原文、``start/end`` 字符范围、``kind`` 内部类型、``source`` 来源、
    ``detail`` 说明。
    输出：候选字典，包含位置、原文、类型、中文标签、来源和说明。

    start/end 是 Python 字符位置，采用左闭右开区间 [start, end)。
    例如“姓名：张三”的姓名范围是 [3, 5)，text[3:5] 恰好是“张三”。
    """
    assert 0 <= start < end <= len(text)
    return dict(start=start, end=end, text=text[start:end], type=kind,
                label=FIELDS[kind], source=source, detail=detail)


def check_id(value):
    """检查身份证格式并生成说明。

    输入：``value`` 身份证字符串。
    输出：中文校验说明字符串；不联网核验真实身份，也不会删除候选。
    """
    if len(value) != 18:
        return '旧版号码候选，待核验'
    try:
        # 第 7～14 位是出生日期；先检查是否为真实日历日期。
        datetime.strptime(value[6:14], '%Y%m%d')
        # 前 17 位分别乘权重，和对 11 取余，再查表得到应有的末位校验码。
        code = '10X98765432'[sum(int(x)*w for x,w in zip(value[:17], [7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2])) % 11]
        if code == value[-1].upper():
            return '格式、生日与校验位通过；未验证证件真实性'
    except ValueError:
        pass
    return '格式候选，校验未通过；仍保护，避免错字造成漏检'


def detect_rules(text):
    """用正则扫描整批原文。

    输入：``text`` 原文字符串。
    输出：候选字典列表；每项含 ``start/end/text/type/label/source/detail``。
    函数不替换文字、不清理 OCR，坐标始终对应输入原文。
    """
    found = []
    full_date_ranges=[]
    ranges=field_ranges(text)
    def pattern(kind, expr, group=0, detail='格式/字段上下文规则', field=False):
        # group=0 取完整匹配；group=1 只取字段值，不包含标签和冒号。
        # field=True 时先用已知标签限制范围，防止空字段吃掉下一字段。
        # 字段规则只在本字段范围内匹配；身份证/电话等独立格式仍扫描全文。
        regex=re.compile(expr,re.MULTILINE)
        matches=(m for start,end in ranges for m in regex.finditer(text,start,end)) if field else regex.finditer(text)
        for m in matches:
            if kind=='DATE':
                # 用捕获组分别读取日期和时间，不能用 findall 拆数字：
                # 连写的“日05 + 小时05”在原文中是0505，不能当成第505天。
                parts=m.groupdict()
                if parts.get('year'):
                    # 包括校验失败的完整日期，避免随后把其中的月日当作有效的无年份日期。
                    full_date_ranges.append(m.span())
                else:
                    if any(m.start()<end and start<m.end() for start,end in full_date_ranges):
                        continue
                    # 年份格式损坏或分隔超限时，不退化成只遮月日、留下年份。
                    if re.search(r'(?:年[ \t\r\n_\u3000\uff3f]*|\d{4}[ \t\r\n_\u3000\uff3f]+)\Z',text[:m.start()]):
                        continue
                try:
                    # 缺年份只检查月日能否成立：用闰年允许2月29日，不补写或推断原文年份。
                    datetime(int(parts.get('year') or 2000),int(parts['month']),int(parts['day']),
                             int(parts.get('hour') or parts.get('hour_cn') or 0),
                             int(parts.get('minute') or parts.get('minute_cn') or 0),
                             int(parts.get('second') or parts.get('second_cn') or 0))
                except ValueError:
                    continue
            a,b=m.span(group)
            # 仅收缩首尾空白，保留值内部的换行和下划线，位置仍对应原文。
            while a<b and text[a].isspace(): a+=1
            while b>a and text[b-1].isspace(): b-=1
            if a<b: found.append(make_span(text,a,b,kind,'规则',detail))
    # 身份证独立格式：18 位数字/末位 X；校验失败仍保留。
    pattern('ID',r'(?<![0-9A-Za-z])\d{17}[\dXx](?![0-9A-Za-z])')
    # 身份证字段：标签后取 15—18 位号码。
    pattern('ID',r'(?:身份证(?:号码?|件号)?)[：:\s]*([0-9Xx]{15,18})',1,field=True)
    # 手机号：独立匹配 11 位手机号，可带 86/+86。
    pattern('PHONE',r'(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)')
    # 电话字段：支持座机、括号、短横线；号码校验在 pipeline.py 中评分。
    pattern('PHONE',r'(?:联系电话|随访电话|手机号|手机|电话)[：:\s]*((?:\+?86[- ]?)?[\d][\d\-－()（） ]{5,24}\d)',1,field=True)
    # 邮箱：按 用户名@域名.后缀 匹配，不要求标签。
    pattern('EMAIL',r'(?<![A-Za-z0-9_.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9_.-])')
    # 保留带冒号姓名规则；另外支持 OCR 中无冒号但有明确结束边界的姓名字段。
    pattern('PATIENT',r'(?:患者姓名|患者本人[）)]?\s*姓名|(?<!医生)(?<!医师)(?<!联系人)(?<!代理人)(?<!手术者)姓名)[ \t]*[：:][ \t]*([\u4e00-\u9fff·]{2,12})(?=[\s，,。；;]|$)',1,
            detail='带冒号的姓名字段：提取2—12个汉字或间隔点，不跨行取值',field=True)
    # 只放宽明确的“姓名”标签，不把“患者”“医生”等角色词当成无冒号字段。
    # 值前允许空格/制表符/全角空格；值后须有空白、标点或文本结束。
    # 超过12个连续姓名字符时整条不命中，不截取前12个字符。
    pattern('PATIENT',r'(?:患者姓名|患者本人[）)]?[ \t\u3000]*姓名|(?<!医生)(?<!医师)(?<!联系人)(?<!代理人)(?<!手术者)姓名)[ \t\u3000]*([\u4e00-\u9fff·]{2,12})(?=[\s，,。；;、！？!?：:（）()【】\[\]]|\Z)',1,
            detail='无冒号的姓名字段：标签后取2—12个汉字或间隔点，以空白、标点或下一字段为界，不跨行取值',field=True)
    # 医生及签名标签：须有冒号，取 2—8 个汉字/间隔点，不跨行取值。
    pattern('DOCTOR',r'(?:主治或主任医师签名|上级医师签名|住院医师签名|医师签名|医生姓名|医师姓名|医生|随访负责人|手术者签名|手术者姓名|术者|助手|助理护士|记录签名|主持人|责任护士|审核签名|签名|签字|审批者)[ \t]*[：:][ \t]*([\u4e00-\u9fff·]{2,8})(?=[\s，,。；;]|$)',1,
            detail='医生姓名字段须有中英文冒号，取2—8个汉字或间隔点',field=True)
    # 联系人/代理人：须有冒号，取 2—12 个汉字/间隔点。
    pattern('CONTACT',r'(?:联系人姓名|代理人姓名|紧急联系人|联系人|代理人)[ \t]*[：:][ \t]*([\u4e00-\u9fff·]{2,12})(?=[\s，,。；;]|$)',1,
            detail='联系人姓名字段须有中英文冒号，取2—12个汉字或间隔点',field=True)
    # 住址/出生地/籍贯：须有冒号；在下一已知字段或换行、分号、句号、冒号处结束。
    pattern('ADDRESS',r'(?:家庭住址|家庭地址|联系地址|居住地址|住址|地址|出生地|籍贯)[ \t]*[：:][ \t]*([^\n\r；;。：:]+)',1,
            detail='住址、出生地、籍贯等标签后须有中英文冒号，遇下一字段停止，不跨行取值',field=True)
    # 病历/住院号：标签后取 3—31 位字母、数字、短横线，冒号可省略。
    pattern('MRN',r'(?:住院号|病历号|门诊号|就诊号|医保号)[：:\t ]*([A-Za-z0-9][A-Za-z0-9\-]{2,30})',1,field=True)
    # 检查/申请号：格式同病历号，触发标签不同。
    pattern('EXAM',r'(?:申请单号|检查申请号|检查号|报告编号)[：:\t ]*([A-Za-z0-9][A-Za-z0-9\-]{2,30})',1,field=True)
    # 床号：标签后取 1—16 位字母、数字、短横线，冒号可省略。
    pattern('BED',r'(?:床号|床位号)[：:\t ]*([A-Za-z0-9\-]{1,16})',1,field=True)
    # 无字段标签：只有整行名称才按医院类后缀识别，行首尾空白不纳入结果。
    pattern('ORG',r'^[ \t]*([\u4e00-\u9fff]{2,30}(?:医院|卫生院|卫生服务中心))[ \t]*\r?$',1)
    # 有字段标签和冒号：按字段取值，不要求名称后缀，可包含院区、数字等。
    pattern('ORG',r'(?:医院名称|医院)[ \t]*[：:][ \t]*([^\n\r，,；;。：:]+)',1,
            detail='医院名称或医院标签后须有中英文冒号，遇下一字段停止，不要求医院类后缀，不跨行',field=True)
    # 先匹配带年份格式，再匹配无年份月日；统一校验捕获的年月日/时间，保留原文坐标。
    # 1. 完整中文年月日，允许有界的OCR空白和下划线，时间可紧接“日”。
    pattern('DATE',r'(?<!\d)(?P<year>(?:19|20)\d{2})[ \t\r\n_\u3000\uff3f]{0,6}年[ \t\r\n_\u3000\uff3f]{0,6}(?P<month>\d{1,2})[ \t\r\n_\u3000\uff3f]{0,6}月[ \t\r\n_\u3000\uff3f]{0,6}(?P<day>\d{1,2})[ \t\r\n_\u3000\uff3f]{0,6}日(?:[ \t]*(?:(?P<hour>\d{1,2})[:：](?P<minute>\d{1,2})(?:[:：](?P<second>\d{1,2}))?|(?P<hour_cn>\d{1,2})时[ \t]*(?P<minute_cn>\d{1,2})分(?:(?P<second_cn>\d{1,2})秒)?))?(?!\d)',detail='中文年月日：数字和标记间允许最多6个空白或下划线；日期时间校验通过')
    # 2. 常规数字日期；日期与时间以空格/制表符分开。
    pattern('DATE',r'(?<!\d)(?P<year>(?:19|20)\d{2})(?P<date_sep>[-./])(?P<month>\d{1,2})(?P=date_sep)(?P<day>\d{1,2})(?!\d)(?:[ \t]+(?:(?P<hour>\d{1,2})[:：](?P<minute>\d{1,2})(?:[:：](?P<second>\d{1,2}))?|(?P<hour_cn>\d{1,2})时[ \t]*(?P<minute_cn>\d{1,2})分(?:(?P<second_cn>\d{1,2})秒)?))?(?!\d)',detail='数字年月日：横线、斜线或点号分隔，可接空格分隔的时间；日期时间校验通过')
    # 3. OCR日期/时间粘连，强制月份、日、小时均为两位，避免三位数字歧义。
    pattern('DATE',r'(?<!\d)(?P<year>(?:19|20)\d{2})(?P<date_sep>[-./])(?P<month>\d{2})(?P=date_sep)(?P<day>\d{2})(?:(?P<hour>\d{2})[:：](?P<minute>\d{2})(?:[:：](?P<second>\d{2}))?|(?P<hour_cn>\d{2})时[ \t]*(?P<minute_cn>\d{2})分(?:(?P<second_cn>\d{2})秒)?)(?!\d)',detail='日期容错：数字日期与时间连写，按两位日和两位小时拆分；日期时间校验通过')
    # 4. OCR缺“月”字：保留“年/日”，月和日数字间必须有1—6个明确分隔字符。
    pattern('DATE',r'(?<!\d)(?P<year>(?:19|20)\d{2})[ \t\r\n_\u3000\uff3f]{0,6}年[ \t\r\n_\u3000\uff3f]{0,6}(?P<month>\d{1,2})[ \t\r\n_\u3000\uff3f]{1,6}(?P<day>\d{1,2})[ \t\r\n_\u3000\uff3f]{0,6}日(?:[ \t]*(?:(?P<hour>\d{1,2})[:：](?P<minute>\d{1,2})(?:[:：](?P<second>\d{1,2}))?|(?P<hour_cn>\d{1,2})时[ \t]*(?P<minute_cn>\d{1,2})分(?:(?P<second_cn>\d{1,2})秒)?))?(?!\d)',detail='日期容错：缺月字，月日数字有明确分隔并保留年/日标记；日期时间校验通过')
    # 5. 无年份月日，可接时间。区间左右两端各自识别，保留“至”等连接词。
    # 必须有“月/日”，不把09-04等可能的床号/编号猜成日期。
    pattern('DATE',r'(?<![\d年])(?P<month>\d{1,2})[ \t\r\n_\u3000\uff3f]{0,6}月[ \t\r\n_\u3000\uff3f]{0,6}(?P<day>\d{1,2})[ \t\r\n_\u3000\uff3f]{0,6}日(?:[ \t]*(?:(?P<hour>\d{1,2})[:：](?P<minute>\d{1,2})(?:[:：](?P<second>\d{1,2}))?|(?P<hour_cn>\d{1,2})时[ \t]*(?P<minute_cn>\d{1,2})分(?:(?P<second_cn>\d{1,2})秒)?))?(?!\d)',detail='无年份月日，可带时分秒；校验月日和时间，不补年份；区间两端分别识别')
    # 年龄：1—3 位数字加“岁”，允许中间空白。
    pattern('AGE',r'(?<!\d)\d{1,3}\s*岁')
    validated=[]
    for span in found:
        if span['type']=='ID':span['detail']=check_id(span['text'])
        validated.append(span)
    return validated
