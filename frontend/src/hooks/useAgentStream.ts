import { useCallback, useMemo, useRef, useState } from "react";
import { Command } from "@langchain/langgraph-sdk";
import { client, ASSISTANT_ID } from "../lib/langgraphClient";
import {
  abortThread,
  fetchHistoryThread,
  rerunFromScene,
} from "../lib/historyClient";
import type {
  InterruptPayload,
  MultimediaState,
  ReviewDecision,
  RunInput,
  RunStatus,
  TimelineEntry,
} from "../types";

/** 从 values 事件中提取 interrupt（不同 server 版本结构略有差异，做多路兜底） */
function extractInterrupt(chunk: unknown): InterruptPayload | null {
  if (!chunk || typeof chunk !== "object") return null;
  const data = chunk as Record<string, unknown>;

  // 形式 A: { __interrupt__: [{ value: {...} }] }
  const raw = data.__interrupt__ ?? data.interrupt;
  if (!raw) return null;

  const first = Array.isArray(raw) ? raw[0] : raw;
  if (!first) return null;

  if (typeof first === "object") {
    const item = first as Record<string, unknown>;
    const value = "value" in item ? item.value : item;
    if (value && typeof value === "object" && "stage" in value) {
      return value as unknown as InterruptPayload;
    }
  }
  return null;
}

let seq = 0;
const nextId = () => `${Date.now()}-${seq++}`;

export function useAgentStream() {
  const [threadId, setThreadId] = useState<string | null>(null);
  // 【修复】stop/reset 需要在回调里读到"最新的 threadId"。
  // 若直接用 threadId state，stop 会捕获旧值（尤其 startRun 刚 setThreadId
  // 之后立刻点停止时读到 null），导致服务端中止请求打空。
  const threadIdRef = useRef<string | null>(null);
  const setThreadIdSafe = useCallback((id: string | null) => {
    threadIdRef.current = id;
    setThreadId(id);
  }, []);
  const [status, setStatus] = useState<RunStatus>("idle");
  const [state, setState] = useState<MultimediaState>({});
  const [interrupt, setInterrupt] = useState<InterruptPayload | null>(null);
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  // 当前 run 是否处于低干预 / 自动模式（用于时间轴标注「自动」审核节点）
  const [autoMode, setAutoMode] = useState(false);
  /** 历史会话恢复后，图的下一步待执行节点（用于判断能否"继续执行"） */
  const [pendingNodes, setPendingNodes] = useState<string[]>([]);

  /** F-1 批量审核队列：预通过镜头集合（0-based）。命中其审核闸口时自动 resume approve，
   *  无需逐个手动确认；video_review 通过后该镜头出队（lock 选项随决策回传）。 */
  const autoQueueRef = useRef<{ scenes: number[]; lock: boolean; used: number } | null>(null);
  const [autoQueue, setAutoQueue] = useState<number[]>([]);

  /** 会话动作模式：唯一的分流器，决定 UI 该渲染哪一套操作入口
   * - `review`   : 存在待审中断，只允许 resumeRun(Command.resume)
   * - `continue` : 无中断但图仍有待执行节点，只允许 continueRun({input:null})
   * - `none`     : 已完成 / 已中止 / 无明显可执行动作
   * 由 interrupt 与 pendingNodes 派生，从结构上保证两类入口互斥，杜绝"双面板"。
   */
  type SessionActionMode = "review" | "continue" | "none";

  const abortRef = useRef<AbortController | null>(null);

  const pushTimeline = useCallback((node: string, detail?: string) => {
    setTimeline((prev) => {
      // 同一节点连续重复时不重复记录
      if (prev.length && prev[prev.length - 1].node === node) return prev;
      return [...prev, { id: nextId(), node, at: Date.now(), detail }];
    });
  }, []);

  /** 消费一次 stream（新建 run 或 resume 共用） */
  const consume = useCallback(
    async (
      tid: string,
      payload: { input: RunInput | null } | { command: Command },
    ) => {
      setStatus("running");
      setError(null);
      setInterrupt(null);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const stream = client.runs.stream(tid, ASSISTANT_ID, {
          ...payload,
          streamMode: ["values", "updates"],
          signal: controller.signal,
        } as Parameters<typeof client.runs.stream>[2]);

        let sawInterrupt = false;

        for await (const chunk of stream) {
          if (chunk.event === "updates" && chunk.data) {
            // updates 的 key 就是刚执行完的节点名
            for (const nodeName of Object.keys(
              chunk.data as Record<string, unknown>,
            )) {
              if (nodeName.startsWith("__")) continue;
              pushTimeline(nodeName);
            }
          }

          if (chunk.event === "values" && chunk.data) {
            const found = extractInterrupt(chunk.data);
            if (found) {
              sawInterrupt = true;
              setInterrupt(found);
            } else {
              setState((prev) => ({
                ...prev,
                ...(chunk.data as MultimediaState),
              }));
            }
          }

          if (chunk.event === "error") {
            const msg =
              typeof chunk.data === "string"
                ? chunk.data
                : JSON.stringify(chunk.data);
            setError(msg);
            setStatus("error");
            return;
          }
        }

        // 流结束后回读一次权威状态，确保 interrupt / 终态判断准确。
        // 【修复】这层回读只是“增强”，不能因它失败就把一次已成功的运行判成 error。
        // 原先它与 consume 同处一个 try，getState 偶发 5xx 会直接落到外层 catch，
        // 覆盖 status 并把 pendingNodes 留在脏状态，UI 卡死在错误页。
        let snapshot: Awaited<ReturnType<typeof client.threads.getState<MultimediaState>>> | null = null;
        try {
          snapshot = await client.threads.getState<MultimediaState>(tid);
          // 先取值到局部常量再进 setState 回调：TS 无法在闭包里窄化可为 null 的 let
          const snapValues = snapshot.values;
          if (snapValues) {
            setState((prev) => ({ ...prev, ...snapValues }));
          }
        } catch (snapErr) {
          console.warn("[useAgentStream] 终态回读失败，按流内结果判定:", snapErr);
        }

        const pending = (snapshot as { tasks?: Array<{ interrupts?: Array<{ value?: unknown }> }> } | null)
          ?.tasks?.flatMap((t) => t.interrupts ?? []) ?? [];
        const pendingValue = pending.find(
          (i) => i?.value && typeof i.value === "object" && "stage" in (i.value as object),
        );

        if (pendingValue) {
          setInterrupt(pendingValue.value as InterruptPayload);
          setStatus("interrupted");
        } else if (sawInterrupt) {
          setStatus("interrupted");
        } else {
          // 【修复】run 正常跑完时必须清空 pendingNodes。
          // 否则历史会话 loadThread / rerunFrom 写入的 pendingNodes（如 ["advance_scene"]）
          // 会残留，导致 App 的 actionMode 一直判为 "continue"，
          // 已完成任务上方永久挂着「该会话尚未跑完 / 继续执行」面板。
          setPendingNodes([]);
          setStatus("done");
        }

        // F-1：当前中断命中预通过队列 → 自动放行并继续消费流
        const _q = autoQueueRef.current;
        if (pendingValue && _q && _q.used < 80) {
          const pv = pendingValue.value as { scene_index?: number; stage?: string };
          const idx0 = typeof pv.scene_index === "number" ? pv.scene_index - 1 : -1;
          if (idx0 >= 0 && _q.scenes.includes(idx0)) {
            const decision: ReviewDecision = { action: "approve" };
            if (pv.stage === "video_review" && _q.lock) decision.lock = true;
            _q.used += 1;
            if (pv.stage === "video_review") {
              // 该镜头全部闸口走完，出队
              _q.scenes = _q.scenes.filter((x) => x !== idx0);
              setAutoQueue([..._q.scenes]);
            }
            console.info(`[F-1] 自动通过 镜头 ${idx0 + 1} 的 ${pv.stage ?? "review"} 闸口`);
            setStatus("running");
            setInterrupt(null);
            void consume(tid, { command: { resume: decision } as Command });
            return;
          }
        }
      } catch (e) {
        if (controller.signal.aborted) {
          setStatus("idle");
          return;
        }
        const rawMsg = e instanceof Error ? e.message : String(e);
        // 对 LangGraph Server 的 404（会话/助手不存在）做友好包装
        const friendlyMsg = rawMsg.includes("HTTP 404")
          ? `会话或助手不存在，请刷新页面后重试。（原始错误: ${rawMsg}）`
          : rawMsg;
        setError(friendlyMsg);
        setStatus("error");
      } finally {
        abortRef.current = null;
      }
    },
    [pushTimeline],
  );

  /** 启动一次全新的生成任务 */
  const startRun = useCallback(
    async (input: RunInput) => {
      // 【修复】启动新任务前先断开任何残留流，保证同一时刻只有一条活跃 SSE，
      // 否则旧流的 events 会污染新任务的状态与进度条。
      abortRef.current?.abort();
      abortRef.current = null;
      setTimeline([]);
      setState({});
      setPendingNodes([]);
      autoQueueRef.current = null;
      setAutoQueue([]);
      setError(null);
      try {
        const thread = await client.threads.create();
        setThreadIdSafe(thread.thread_id);
        setAutoMode(!!input.auto_mode);
        await consume(thread.thread_id, { input });
        return thread.thread_id;
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") return;
        const msg = e instanceof Error ? e.message : String(e);
        setError(`启动任务失败：${msg}`);
        setStatus("error");
      }
    },
    [consume],
  );

  /** 人审后续跑：Command(resume=decision) */
  const resumeRun = useCallback(
    async (decision: ReviewDecision) => {
      if (!threadId) return;
      try {
        // 前置校验：确认 thread 在 LangGraph Server 中仍然有效
        try {
          await client.threads.getState(threadId);
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          if (
            msg.includes("404") ||
            msg.includes("not found") ||
            msg.includes("Thread or assistant")
          ) {
            setError(
              `该会话已失效，无法恢复执行。请重新创建任务。（会话ID: ${threadId.slice(0, 8)}…）`,
            );
            setStatus("error");
            return;
          }
          throw e;
        }

        await consume(threadId, {
          command: { resume: decision } as Command,
        });
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") return;
        const msg = e instanceof Error ? e.message : String(e);
        setError(`恢复执行失败：${msg}`);
        setStatus("error");
      }
    },
    [consume, threadId],
  );

  /**
   * 加载历史会话（双通道）。
   *
   * 通道 1：自建 /history 接口，直接从 sqlite 读原始 checkpoint。历史会话
   *         （尤其是旧版图跑出来的）只有这条路可靠——平台 getState 会在路由外
   *         抛 500，无法拦截。
   * 通道 2：平台 client.threads.getState，作为通道 1 不可用时的兜底
   *         （如产物服务未启动）。
   *
   * 两条通道都失败才 setError，从根本上消除"点击没反应"。
   */
  const loadThread = useCallback(async (tid: string) => {
    // 【修复】切换历史会话时残留 SSE 流竞态：
    // 若上一个会话的流还在推 events，它会继续 setState/timeline，
    // 与本会话加载结果交叉覆盖（表现为进度条乱跳、状态闪回）。
    // 先把残留流彻底断开并置空 ref，保证任意时刻只有一条活跃流。
    abortRef.current?.abort();
    abortRef.current = null;

    setThreadIdSafe(tid);
    setTimeline([]);
    setError(null);
    setInterrupt(null);
    setPendingNodes([]);
    autoQueueRef.current = null;
    setAutoQueue([]);

    // —— 通道 1：自建历史接口 ——
    try {
      const detail = await fetchHistoryThread(tid);
      setState(detail.values ?? {});
      // 【修复】过滤 __start__ / __end__ 等内部节点。
      // 已完成任务的 checkpoint.next 是 ["__end__"]，不过滤会让 App 把
      // 已完成任务判成"尚未跑完"，并把内部节点名直接显示给用户。
      setPendingNodes(
        (detail.progress?.next ?? []).filter(
          (n): n is string => !!n && !n.startsWith("__"),
        ),
      );
      // 【修复】历史会话进度时间轴全空：
      // 以前 loadThread 只 setPendingNodes，历史已完成任务的 next 为空，
      // 进度条直接空白、看不出跑过哪些节点。后端现已从 checkpoint 回放
      // executed_nodes，用它重建时间轴（executed=已完成, next=待跑）。
      const executed = (detail.progress?.executed_nodes ?? []).filter(
        (n): n is string => !!n && !n.startsWith("__"),
      );
      if (executed.length > 0) {
        // 字段名须与 TimelineEntry 一致：{ id, node, at, detail }
        setTimeline(
          executed.map((n) => ({
            id: `history-${n}`,
            node: n,
            at: Date.now(),
            detail: "历史记录回放",
          })),
        );
      } else {
        setTimeline([]);
      }

      if (detail.interrupt) {
        setInterrupt(detail.interrupt);
        setStatus("interrupted");
      } else if (detail.status === "done") {
        setStatus("done");
      } else if (detail.status === "aborted" || detail.status === "corrupted") {
        setStatus("error");
        setError(
          detail.status === "aborted"
            ? `该会话已中止${
                detail.values?.abort_reason
                  ? `：${detail.values.abort_reason}`
                  : ""
              }`
            : "该会话的存档已损坏，无法恢复",
        );
      } else {
        setStatus("idle");
      }
      return;
    } catch (historyErr) {
      // —— 通道 2：平台 getState 兜底 ——
      try {
        const snapshot = await client.threads.getState<MultimediaState>(tid);
        setState(snapshot.values ?? {});
        setPendingNodes(
          ((snapshot as { next?: string[] }).next ?? []).filter(
            (n): n is string => !!n && !n.startsWith("__"),
          ),
        );

        const pending =
          (snapshot as { tasks?: Array<{ interrupts?: Array<{ value?: unknown }> }> })
            .tasks?.flatMap((t) => t.interrupts ?? []) ?? [];
        const pendingValue = pending.find(
          (i) => i?.value && typeof i.value === "object" && "stage" in (i.value as object),
        );

        if (pendingValue) {
          setInterrupt(pendingValue.value as InterruptPayload);
          setStatus("interrupted");
        } else {
          setStatus(snapshot.values ? "done" : "idle");
        }
        return;
      } catch (platformErr) {
        const msg =
          historyErr instanceof Error ? historyErr.message : String(historyErr);
        const fallbackMsg =
          platformErr instanceof Error
            ? platformErr.message
            : String(platformErr);
        // 两条通道都失败时，区分「服务未启动」与「数据/接口异常」，给出可操作提示
        const serviceDown = /连接产物服务|无法连接|fetch/i.test(msg + fallbackMsg);
        const hint = serviceDown
          ? "请确认已通过一键启动脚本或手动启动全部服务（LangGraph 2024 + 静态服务 8900 + 前端 5173），并刷新页面后重试。"
          : "若服务已启动仍无法加载，请查看后端日志（static_server.py / langgraph）排查历史存档数据。";
        setError(
          `无法加载该历史会话（${tid.slice(0, 8)}…）：${msg}（备用通道亦失败：${fallbackMsg}）。${hint}`,
        );
        setStatus("error");
        setState({});
      }
    }
  }, []);

  /**
   * 继续执行历史会话。
   *
   * 无中断时以 `{ input: null }` 让图从最新 checkpoint 继续推进
   * （LangGraph 标准续跑语义）；有中断的场景请走 resumeRun。
   */
  const continueRun = useCallback(async () => {
    if (!threadId) return;
    try {
      await consume(threadId, { input: null });
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      const msg = e instanceof Error ? e.message : String(e);
      setError(`继续执行失败：${msg}`);
      setStatus("error");
    }
  }, [consume, threadId]);

  /** 会话动作模式：唯一分流器，保证审核卡片与续跑面板互斥 */
  const actionMode: SessionActionMode = useMemo(() => {
    if (interrupt) return "review";
    if (!!threadId && pendingNodes.length > 0 && status !== "running") return "continue";
    return "none";
  }, [interrupt, threadId, pendingNodes.length, status]);

  /**
   * 从指定镜头重新跑：先让后端把该 thread 的 checkpoint 重置到目标镜头
   * （current_scene_index + 清空后续产物 + next=advance_scene），再以
   * { input: null } 续跑，图会从该镜头的 visual_context_builder 重头制作。
   */
  const rerunFrom = useCallback(
    async (sceneIndex: number, opts?: { unlock?: boolean }) => {
      if (!threadId) return;
      setError(null);
      try {
        const r = await rerunFromScene(threadId, sceneIndex, opts?.unlock ?? false);
        // 重置前端视图，让续跑流重新填充状态
        setState((prev) => ({
          ...prev,
          current_scene_index: r.scene_index,
          final_movie_path: null,
          final_movie_with_audio: null,
          error_log: null,
          aborted: false,
          abort_reason: null,
        }));
        setPendingNodes(["advance_scene"]);
        await consume(threadId, { input: null });
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") return;
        const msg = e instanceof Error ? e.message : String(e);
        setError(`从镜头 ${sceneIndex + 1} 重新跑失败：${msg}`);
        setStatus("error");
      }
    },
    [consume, threadId],
  );

  const stop = useCallback(() => {
    // 1) 先断前端流（同步、立即生效）
    abortRef.current?.abort();
    // 2) 再通知后端把该 thread 标记为 aborted。
    //    否则后端 run 仍在跑，用户再次启动会对同一 thread 并发跑两条 run
    //    互相覆盖 state（产物错乱/进度回跳）。失败也不阻塞（尽力而为）。
    if (threadIdRef.current) {
      void abortThread(threadIdRef.current);
    }
  }, []);

  /** 新建会话：中断当前运行并清空所有会话状态，回到干净的创建态 */
  const reset = useCallback(() => {
    // 复用 stop 的语义：断流 + 通知后端中止（避免残留 run 与新会话并发）
    if (threadIdRef.current) {
      void abortThread(threadIdRef.current);
    }
    abortRef.current?.abort();
    abortRef.current = null;
    setThreadIdSafe(null);
    setState({});
    setInterrupt(null);
    setTimeline([]);
    setError(null);
    setPendingNodes([]);
    autoQueueRef.current = null;
    setAutoQueue([]);
    setAutoMode(false);
    setStatus("idle");
  }, []);

  return {
    threadId,
    status,
    state,
    interrupt,
    timeline,
    error,
    pendingNodes,
    autoMode,
    actionMode,
    startRun,
    resumeRun,
    continueRun,
    rerunFrom,
    autoQueue,
    planAutoApprove: (scenes: number[], lock: boolean) => {
      autoQueueRef.current = { scenes: [...scenes], lock, used: 0 };
      setAutoQueue([...scenes]);
    },
    cancelAutoApprove: () => {
      autoQueueRef.current = null;
      setAutoQueue([]);
    },
    loadThread,
    stop,
    reset,
  };
}
