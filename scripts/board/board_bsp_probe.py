# -*- coding: utf-8 -*-
"""
RK3588 板端「系统层 / BSP 认知」只读探查

用法：sudo python3 board_bsp_probe.py

性质：纯只读。不写 sysfs、不改设备树、不加载/卸载驱动、不重转模型。
      缺工具会标注 [空 / 命令不存在]，任何一项失败都不影响其余项。
目的：把「会用 dmesg/devfreq 排障」升级成「讲得清设备树 → 驱动 → 内存 →
      时钟电源 → 设备节点」这条因果链，用于面试追问与项目笔记。
"""

import os
import re
import glob
import platform
import subprocess

DT = '/proc/device-tree'


def sh(cmd, timeout=10, limit=40):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, timeout=timeout)
        out = (r.stdout or '') + (r.stderr or '')
    except subprocess.TimeoutExpired:
        return '    [超时]'
    except Exception as e:
        return '    [执行失败: %s]' % e
    lines = [l.rstrip() for l in out.splitlines() if l.strip()]
    if not lines:
        return '    [空 / 命令不存在]'
    if len(lines) > limit:
        lines = lines[:limit] + ['    ... 共 %d 行，已截断' % len(lines)]
    return '\n'.join('    ' + l for l in lines)


def section(n, title):
    print('\n' + '=' * 64)
    print(' [%s] %s' % (n, title))
    print('=' * 64)


def item(label, cmd, **kw):
    print('\n  -- %s' % label)
    print(sh(cmd, **kw))


def read_attr(path):
    try:
        with open(path, 'rb') as f:
            b = f.read()
    except Exception as e:
        return '<读不到: %s>' % e
    if not b:
        return '<空>'
    s = b.replace(b'\x00', b' ').decode('utf-8', 'replace').strip()
    printable = sum(1 for c in s if c.isprintable())
    if s and printable >= len(s) * 0.8:
        return repr(s)
    h = b.hex()
    return h[:64] + ('...' if len(h) > 64 else '')


def probe_device_tree():
    print('\n  -- 平台标识')
    for name in ('model', 'compatible', 'serial-number'):
        p = os.path.join(DT, name)
        print('    %-16s %s' % (name, read_attr(p) if os.path.exists(p) else '<不存在>'))

    print('\n  -- 名字含 npu / fdab0000 的设备树节点')
    if not os.path.isdir(DT):
        print('    [ /proc/device-tree 不存在，内核未导出 dtb ]')
        return
    hits = []
    for root, dirs, files in os.walk(DT):
        if re.search(r'npu|fdab0000', root, re.I):
            hits.append(root)
    if not hits:
        print('    [ 未找到 ]')
        return
    for h in hits[:8]:
        rel = h.replace(DT, '') or '/'
        print('    节点 %s' % rel)
        for f in sorted(os.listdir(h))[:24]:
            fp = os.path.join(h, f)
            if os.path.isfile(fp):
                print('      %-24s %s' % (f, read_attr(fp)))
    if len(hits) > 8:
        print('    ... 另有 %d 个节点未展开' % (len(hits) - 8))


print('=' * 64)
print(' RK3588 板级系统层只读探查')
print(' 内核 %s | 架构 %s | UID %s' % (platform.release(), platform.machine(), os.geteuid()))
print(' 本脚本不修改任何系统状态')
print('=' * 64)

section(1, '系统 / 内核 / 启动链')
item('uname -a', 'uname -a')
item('/proc/version', 'cat /proc/version')
item('发行版', "grep -E '^(NAME|VERSION)=' /etc/os-release")
item('内核 cmdline（U-Boot 传参）', 'cat /proc/cmdline')
item('内核配置里的 RKNPU（m=模块 / y=内置）',
     "zcat /proc/config.gz 2>/dev/null | grep -i rknpu; "
     "for f in /boot/config-*; do [ -f \"$f\" ] && grep -i rknpu \"$f\"; done", limit=20)
item('启动阶段 rknpu/drm/iommu 时间戳',
     "dmesg | grep -iE 'rknpu|drm|iommu' | head -25", limit=25)

section(2, '设备树（BSP 最核心的一层）')
probe_device_tree()
item('dtc 导出 npu 节点片段（需 device-tree-compiler）',
     "dtc -I fs -O dts /proc/device-tree 2>/dev/null | grep -A 30 -i 'npu@fdab0000' | head -35",
     limit=35)

section(3, '驱动形态 / 设备节点')
item('debugfs 里的驱动版本', 'cat /sys/kernel/debug/rknpu/version 2>/dev/null')
item('debugfs 里的 NPU 负载', 'cat /sys/kernel/debug/rknpu/load 2>/dev/null')
item('驱动是模块还是 builtin',
     "lsmod | grep -i rknpu; ls -1 /sys/module/ 2>/dev/null | grep -i rknpu; "
     "grep -i rknpu /proc/devices")
item('字符设备节点（老 BSP 形态）', 'ls -l /dev/rknpu* 2>/dev/null; ls -1 /sys/class/misc/ | grep -i rknpu')
item('DRM render node（主线形态）', 'ls -l /dev/dri/ 2>/dev/null')
item('DRM device 与 iommu 关联',
     "dmesg | grep -iE 'renderD|gem name|iommu domain' | tail -15", limit=15)
item('renderD129 权限与主从设备号',
     "stat -c '%A %U:%G  major=%t minor=%T' /dev/dri/renderD129 2>/dev/null; "
     "id; getent group render video")
item('debugfs 挂载情况', "mount | grep -E 'debugfs|tracefs'")

section(4, '内存 / IOMMU / DMA')
item('CMA 预留与 meminfo', "dmesg | grep -i 'cma: Reserved'; grep -i cma /proc/meminfo")
item('iommu debugfs 组', "find /sys/kernel/debug/iommu -maxdepth 2 -type d 2>/dev/null | head -20", limit=20)
item('DMA / iommu 相关启动日志', "dmesg | grep -iE 'iommu|dma-buf|dma_alloc|swiotlb' | head -20", limit=20)
item('运行时库 librknnrt.so', "ls -l /usr/lib/librknnrt.so 2>/dev/null; "
     "strings /usr/lib/librknnrt.so 2>/dev/null | grep -oE '/dev/(dri|rknpu)[^ ]*' | sort -u | head -10")

section(5, '时钟 / 电源域 / 频率')
item('NPU devfreq 节点', "ls -1 /sys/class/devfreq/ 2>/dev/null")
for a in ('governor', 'cur_freq', 'min_freq', 'max_freq',
          'available_frequencies', 'available_governors'):
    item('fdab0000.npu / %s' % a,
         'cat /sys/class/devfreq/fdab0000.npu/%s 2>/dev/null' % a, limit=8)
item('clk_summary 里的 npu 时钟',
     "head -1 /sys/kernel/debug/clk/clk_summary 2>/dev/null; "
     "grep -i npu /sys/kernel/debug/clk/clk_summary 2>/dev/null | head -20", limit=22)
item('power domain 摘要里的 npu',
     "grep -iE 'npu|domain' /sys/kernel/debug/pm_genpd/pm_genpd_summary 2>/dev/null | head -20",
     limit=20)
item('CPU cpufreq 各 policy',
     "for p in /sys/devices/system/cpu/cpufreq/policy*; do "
     "printf '%s  ' \"$(basename $p)\"; "
     "printf 'gov=%s  cur=%s  max=%s  cpus=%s\\n' "
     "\"$(cat $p/scaling_governor 2>/dev/null)\" "
     "\"$(cat $p/cpuinfo_cur_freq 2>/dev/null)\" "
     "\"$(cat $p/cpuinfo_max_freq 2>/dev/null)\" "
     "\"$(cat $p/affected_cpus 2>/dev/null)\"; done", limit=16)

section(6, 'V4L2 / media controller 拓扑')
item('video4linux 节点数', "ls -1 /sys/class/video4linux/ 2>/dev/null | wc -l")
item('前 12 个 video 节点的名字',
     "for v in $(ls -1 /sys/class/video4linux/ | head -12); do "
     "printf '%-10s ' \"$v\"; cat /sys/class/video4linux/$v/name 2>/dev/null; done", limit=14)
item('media-ctl 是否可用', 'which media-ctl v4l2-ctl 2>/dev/null')
item('media0 拓扑（前 55 行）', "media-ctl -p -d /dev/media0 2>&1 | head -55", limit=55)
item('v4l2-ctl --list-devices', "v4l2-ctl --list-devices 2>&1 | head -35", limit=35)
item('sensor / isp / cif 相关日志',
     "dmesg | grep -iE 'sensor|rkcif|rkisp|remote terminal' | tail -20", limit=20)

section(7, '版本链（用户态 → 驱动）')
item('rknnlite 版本',
     "python3 -c \"from rknnlite.api import RKNNLite; "
     "r=RKNNLite(); print(getattr(r,'get_sdk_version',lambda: 'n/a')())\" 2>&1 | tail -5",
     limit=6)
item('toolkit / driver / 系统库一致性线索',
     "cat /sys/kernel/debug/rknpu/version 2>/dev/null; "
     "strings /usr/lib/librknnrt.so 2>/dev/null | grep -oE '[0-9]+\\.[0-9]+\\.[0-9]+' | sort -u | head -8")

print('\n' + '=' * 64)
print(' 探查结束。以上全为只读结果，未修改任何系统状态。')
print(' 把输出贴回来即可生成《板级系统与驱动认知笔记》。')
print('=' * 64)
