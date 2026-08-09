import { useCallback, useRef, useState } from "react";
import { Command } from "@langchain/langgraph-sdk";
import { client, ASSISTANT_ID } from "../lib/langgraphClient";
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
  const [status, setStatus] = useState<RunStatus>("idle");
  const [state, setState] = useState<MultimediaState>({});
  const [interrupt, setInterrupt] = useState<InterruptPayload | null>(null);
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

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

        // 流结束后回读一次权威状态，确保 interrupt / 终态判断准确
        const snapshot = await client.threads.getState<MultimediaState>(tid);
        if (snapshot.values) {
          setState((prev) => ({ ...prev, ...snapshot.values }));
        }

        const pending = (snapshot as { tasks?: Array<{ interrupts?: Array<{ value?: unknown }> }> })
          .tasks?.flatMap((t) => t.interrupts ?? []) ?? [];
        const pendingValue = pending.find(
          (i) => i?.value && typeof i.value === "object" && "stage" in (i.value as object),
        );

        if (pendingValue) {
          setInterrupt(pendingValue.value as InterruptPayload);
          setStatus("interrupted");
        } else if (sawInterrupt) {
          setStatus("interrupted");
        } else {
          setStatus("done");
        }
      } catch (e) {
        if (controller.signal.aborted) {
          setStatus("idle");
          return;
        }
        setError(e instanceof Error ? e.message : String(e));
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
      setTimeline([]);
      setState({});
      const thread = await client.threads.create();
      setThreadId(thread.thread_id);
      await consume(thread.thread_id, { input });
      return thread.thread_id;
    },
    [consume],
  );

  /** 人审后续跑：Command(resume=decision) */
  const resumeRun = useCallback(
    async (decision: ReviewDecision) => {
      if (!threadId) return;
      await consume(threadId, {
        command: { resume: decision } as Command,
      });
    },
    [consume, threadId],
  );

  /** 加载历史会话 */
  const loadThread = useCallback(async (tid: string) => {
    setThreadId(tid);
    setTimeline([]);
    setError(null);
    const snapshot = await client.threads.getState<MultimediaState>(tid);
    setState(snapshot.values ?? {});

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
      setInterrupt(null);
      setStatus(snapshot.values ? "done" : "idle");
    }
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return {
    threadId,
    status,
    state,
    interrupt,
    timeline,
    error,
    startRun,
    resumeRun,
    loadThread,
    stop,
  };
}
