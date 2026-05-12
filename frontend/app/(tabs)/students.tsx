import { EmptyState, AppScreen, SectionHeader } from '../../components/AppUI';

export default function StudentsScreen() {
  return (
    <AppScreen>
      <SectionHeader
        eyebrow="Giáo viên"
        title="Danh sách học viên"
        subtitle="Danh sách thật sẽ được tải từ lớp học khi backend cung cấp dữ liệu."
      />
      <EmptyState
        title="Chưa có học viên để hiển thị"
        message="Khi lớp học có dữ liệu, danh sách học viên sẽ xuất hiện tại đây."
      />
    </AppScreen>
  );
}
