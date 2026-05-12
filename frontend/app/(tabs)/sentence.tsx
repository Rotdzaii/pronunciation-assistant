import { EmptyState, AppScreen, SectionHeader } from '../../components/AppUI';
import { BackButton } from '../../components/BackButton';

export default function SentenceScreen() {
  return (
    <AppScreen>
      <BackButton label="Quay lại chọn chế độ" fallbackHref="/(tabs)/practice-mode" />
      <SectionHeader
        eyebrow="Luyện câu"
        title="Luyện câu"
        subtitle="Không gian luyện câu ngắn sẽ dùng chung luồng ghi âm và chấm phát âm."
      />
      <EmptyState
        title="Chưa có câu luyện tập"
        message="Hãy chọn một bài luyện để bắt đầu."
      />
    </AppScreen>
  );
}
