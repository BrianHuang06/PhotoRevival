import { useState } from "react";
import { useAppStore } from "@/store/useAppStore";
import TaskCard from "@/components/TaskCard/TaskCard";
import { Search, Trash2, Inbox } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";

export default function History() {
  const { history, clearHistory } = useAppStore();
  const [search, setSearch] = useState("");

  const filtered = history.filter((t) =>
    t.filename.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="mx-auto h-[calc(100vh-3.5rem)] max-w-6xl overflow-auto p-6">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="font-display text-2xl font-semibold text-ivory">
            历史记录
          </h2>
          <p className="mt-1 text-sm text-ivory-dim">
            共 {history.length} 条修复记录
          </p>
        </div>
        {history.length > 0 && (
          <button
            onClick={clearHistory}
            className="flex items-center gap-1.5 rounded-lg border border-dark-600 bg-dark-800 px-3 py-2 text-xs font-medium text-ivory-dim transition-colors hover:border-red-500/30 hover:bg-red-500/10 hover:text-red-400"
          >
            <Trash2 className="h-3.5 w-3.5" />
            清空全部
          </button>
        )}
      </div>

      {history.length > 0 && (
        <div className="relative mb-6">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ivory-dim" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="搜索文件名..."
            className="w-full rounded-xl border border-dark-700/50 bg-dark-800/50 py-2.5 pl-10 pr-4 text-sm text-ivory outline-none transition-colors placeholder:text-ivory-dim focus:border-amber-accent/30"
          />
        </div>
      )}

      {filtered.length > 0 ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          <AnimatePresence>
            {filtered.map((task) => (
              <TaskCard key={task.id} task={task} />
            ))}
          </AnimatePresence>
        </div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center justify-center py-32"
        >
          <Inbox className="mb-4 h-16 w-16 text-dark-600" />
          <p className="text-lg font-medium text-ivory-muted">暂无修复记录</p>
          <p className="mt-1 text-sm text-ivory-dim">
            前往工作台上传照片并开始修复
          </p>
        </motion.div>
      )}
    </div>
  );
}
